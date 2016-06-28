 /*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef GRAPH_H
#define GRAPH_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <malloc.h>
#include <omp.h>
#include <string.h>
#include <sched.h>
#include <numa.h>

#include <thread>
#include <vector>

#include "core/constants.hpp"
#include "core/type.hpp"
#include "core/bitmap.hpp"
#include "core/atomic.hpp"
#include "core/queue.hpp"
#include "core/partition.hpp"
#include "core/bigvector.hpp"
#include "core/time.hpp"

bool f_true(VertexId v) {
	return true;
}

void f_none_1(std::pair<VertexId,VertexId> vid_range) {

}

void f_none_2(std::pair<VertexId,VertexId> source_vid_range, std::pair<VertexId,VertexId> target_vid_range) {

}

class Graph {
	int parallelism;
	int edge_unit;
	int record_col;
	bool * should_access_shard;
	bool * should_access_shard_col;
	bool ** should_access_shard_rc;
	long ** fsize;
	char ** buffer_pool;
	long * column_offset1;
	long * column_offset2;
	long * row_offset;
	long memory_bytes;
	int partition_batch;
	long vertex_data_bytes;
	long PAGESIZE;
	int col_del;
public:
	std::string path;

	int edge_type;
	VertexId vertices;
	EdgeId edges;
	int partitions;

	Graph (std::string path,int col) {
		PAGESIZE = 4096;
		parallelism = std::thread::hardware_concurrency();
		col_del=col;
		//parallelism = 16;
		buffer_pool = new char * [parallelism*1];
		for (int i=0;i<parallelism*1;i++) {
			buffer_pool[i] = (char *)memalign(PAGESIZE, IOSIZE);
			assert(buffer_pool[i]!=NULL);
			memset(buffer_pool[i], 0, IOSIZE);
		}
		init(path);
	}

	void set_memory_bytes(long memory_bytes) {
		this->memory_bytes = memory_bytes;
	}

	void set_vertex_data_bytes(long vertex_data_bytes) {
		this->vertex_data_bytes = vertex_data_bytes;
	}

	void init(std::string path) {
		this->path = path;

		FILE * fin_meta = fopen((path+"/meta").c_str(), "r");
		fscanf(fin_meta, "%d %d %ld %d %d", &edge_type, &vertices, &edges, &partitions, &record_col);
		fclose(fin_meta);

		if (edge_type==0) {
			PAGESIZE = 4096;
		} else {
			PAGESIZE = 12288;
		}

		should_access_shard = new bool[partitions];
		should_access_shard_col = new bool[partitions];
		should_access_shard_rc =new bool* [partitions];
		for(int i=0;i<partitions;i++)
			should_access_shard_rc[i]=new bool [partitions];

		if (edge_type==0) {
			edge_unit = sizeof(VertexId) * 2;
		} else {
			edge_unit = sizeof(VertexId) * 2 + sizeof(Weight);
		}

		memory_bytes = 1024l*1024l*1024l*1024l; // assume RAM capacity is very large
		partition_batch = partitions;
		vertex_data_bytes = 0;

		char filename[1024];
		fsize = new long * [partitions];
		for (int i=0;i<partitions;i++) {
			fsize[i] = new long [partitions];
			for (int j=0;j<partitions;j++) {
				sprintf(filename, "%s/block-%d-%d", path.c_str(), i, j);
				fsize[i][j] = file_size(filename);
			}
		}

		long bytes;

		column_offset1 = new long [record_col*partitions+1];
		int fin_column_offset1 = open((path+"/column_offset-1").c_str(), O_RDONLY);
		bytes = read(fin_column_offset1, column_offset1, sizeof(long)*(record_col*partitions+1));
		assert(bytes==sizeof(long)*(record_col*partitions+1));
		close(fin_column_offset1);
		
		column_offset2 = new long [partitions*(partitions-record_col)+1];
		int fin_column_offset2 = open((path+"/column_offset-2").c_str(), O_RDONLY);
		bytes = read(fin_column_offset2, column_offset2, sizeof(long)*(partitions*(partitions-record_col)+1));
		assert(bytes==sizeof(long)*(partitions*(partitions-record_col)+1));
		close(fin_column_offset2);		

		row_offset = new long [partitions*partitions+1];
		int fin_row_offset = open((path+"/row_offset").c_str(), O_RDONLY);
		bytes = read(fin_row_offset, row_offset, sizeof(long)*(partitions*partitions+1));
		assert(bytes==sizeof(long)*(partitions*partitions+1));
		close(fin_row_offset);
	}

	Bitmap * alloc_bitmap() {
		return new Bitmap(vertices);
	}

	template <typename T>
	T stream_vertices(std::function<T(VertexId)> process, Bitmap * bitmap = nullptr, T zero = 0,
		std::function<void(std::pair<VertexId,VertexId>)> pre = f_none_1,
		std::function<void(std::pair<VertexId,VertexId>)> post = f_none_1) {
		T value = zero;
		if (bitmap==nullptr && vertex_data_bytes > (0.8 * memory_bytes)) {
			for (int cur_partition=0;cur_partition<partitions;cur_partition+=partition_batch) {
				VertexId begin_vid, end_vid;
				begin_vid = get_partition_range(vertices, partitions, cur_partition).first;
				if (cur_partition+partition_batch>=partitions) {
					end_vid = vertices;
				} else {
					end_vid = get_partition_range(vertices, partitions, cur_partition+partition_batch).first;
				}
				pre(std::make_pair(begin_vid, end_vid));
				#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
				for (int partition_id=cur_partition;partition_id<cur_partition+partition_batch;partition_id++) {
					if (partition_id < partitions) {
						T local_value = zero;
						VertexId begin_vid, end_vid;
						std::tie(begin_vid, end_vid) = get_partition_range(vertices, partitions, partition_id);
						for (VertexId i=begin_vid;i<end_vid;i++) {
							local_value += process(i);
						}
						write_add(&value, local_value);
					}
				}
				#pragma omp barrier
				post(std::make_pair(begin_vid, end_vid));
			}
		} else {
			#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
			for (int partition_id=0;partition_id<partitions;partition_id++) {
				T local_value = zero;
				VertexId begin_vid, end_vid;
				std::tie(begin_vid, end_vid) = get_partition_range(vertices, partitions, partition_id);
				if (bitmap==nullptr) {
					for (VertexId i=begin_vid;i<end_vid;i++) {
						local_value += process(i);
					}
				} else {
					VertexId i = begin_vid;
					while (i<end_vid) {
						unsigned long word = bitmap->data[WORD_OFFSET(i)];
						if (word==0) {
							i = (WORD_OFFSET(i) + 1) << 6;
							continue;
						}
						size_t j = BIT_OFFSET(i);
						word = word >> j;
						while (word!=0) {
							if (word & 1) {
								local_value += process(i);
							}
							i++;
							j++;
							word = word >> 1;
							if (i==end_vid) break;
						}
						i += (64 - j);
					}
				}
				write_add(&value, local_value);
			}
			#pragma omp barrier
		}
		return value;
	}

	void set_partition_batch(long bytes) {
		//int x = (int)ceil(bytes / (0.8 * memory_bytes));
		//partition_batch = partitions / x;
		partition_batch=partitions;
	}

	template <typename... Args>
	void hint(Args... args);

	template <typename A>
	void hint(BigVector<A> & a) {
		long bytes = sizeof(A) * a.length;
		set_partition_batch(bytes);
	}

	template <typename A, typename B>
	void hint(BigVector<A> & a, BigVector<B> & b) {
		long bytes = sizeof(A) * a.length + sizeof(B) * b.length;
		set_partition_batch(bytes);
	}

	template <typename A, typename B, typename C>
	void hint(BigVector<A> & a, BigVector<B> & b, BigVector<C> & c) {
		long bytes = sizeof(A) * a.length + sizeof(B) * b.length + sizeof(C) * c.length;
		set_partition_batch(bytes);
	}

	template <typename T>
	T stream_edges(std::function<T(Edge&)> process, Bitmap * bitmap = nullptr, T zero = 0, int update_mode = 1,
		std::function<void(std::pair<VertexId,VertexId> vid_range)> pre_source_window = f_none_1,
		std::function<void(std::pair<VertexId,VertexId> vid_range)> post_source_window = f_none_1,
		std::function<void(std::pair<VertexId,VertexId> vid_range)> pre_target_window = f_none_1,
		std::function<void(std::pair<VertexId,VertexId> vid_range)> post_target_window = f_none_1) {
		
		//printf("memory_bytes: %ld\n",memory_bytes);
		//printf("memory bytes: %")
		if(col_del==0){
			//printf("flag0\n");
			if (bitmap==nullptr) {
				for (int i=0;i<partitions;i++) {
					should_access_shard[i] = true;
					should_access_shard_col[i]=true;
					
					for(int j=0;j<partitions;j++)
						should_access_shard_rc[i][j]=true;
				}
			} else {
				for (int i=0;i<partitions;i++) {
					should_access_shard[i] = false;
					should_access_shard_col[i]=true;
					
					for(int j=0;j<partitions;j++)
						should_access_shard_rc[i][j]=false;
				}
				#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
				for (int partition_id=0;partition_id<partitions;partition_id++) {
					VertexId begin_vid, end_vid;
					std::tie(begin_vid, end_vid) = get_partition_range(vertices, partitions, partition_id);
					VertexId i = begin_vid;
					while (i<end_vid) {
						unsigned long word = bitmap->data[WORD_OFFSET(i)];
						if (word!=0) {
							should_access_shard[partition_id] = true;
							
							for(int k=0;k<partitions;k++)
								should_access_shard_rc[partition_id][k]=true;
							
							break;
						}
						i = (WORD_OFFSET(i) + 1) << 6;
					}
				}
				#pragma omp barrier
			}
		}
		else if(col_del==1){
			
			if (bitmap==nullptr) {
				for (int i=0;i<partitions;i++) {
					should_access_shard[i] = true;
					should_access_shard_col[i]=true;
					
					for(int j=0;j<partitions;j++)
						should_access_shard_rc[i][j]=true;
				}
			} else {
				
				int r=open((path+"/record_c").c_str(),O_RDONLY);
				unsigned long * f_map=(unsigned long *)mmap(NULL,vertices*sizeof(unsigned long),PROT_READ,MAP_SHARED,r,0);
				struct bitmask *bmask=numa_allocate_nodemask();
				numa_bitmask_setall(bmask);
				numa_tonodemask_memory(f_map,vertices*sizeof(unsigned long),bmask);
				
				for (int i=0;i<partitions;i++) {
					should_access_shard[i] = false;
					should_access_shard_col[i]=false;
					
					for(int j=0;j<partitions;j++)
						should_access_shard_rc[i][j]=false;
				}
				#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
				for (int partition_id=0;partition_id<partitions;partition_id++) {
					VertexId begin_vid, end_vid;
					std::tie(begin_vid, end_vid) = get_partition_range(vertices, partitions, partition_id);
					VertexId i = begin_vid;
					while (i<end_vid) {
						unsigned long word = bitmap->data[WORD_OFFSET(i)];
						if (word!=0) {
							should_access_shard[partition_id] = true;
							break;
						}
						i = (WORD_OFFSET(i) + 1) << 6;
					}
				}
				#pragma omp barrier
						
				#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
				for(int partition_id=0;partition_id<partitions;partition_id++){
					unsigned long mask=0;
					__sync_fetch_and_or(&mask,1<<partition_id);
					for(int p_id=0;p_id<partitions;p_id++){
						if(should_access_shard[p_id]){						
							VertexId begin_vid,end_vid;
							std::tie(begin_vid,end_vid)=get_partition_range(vertices,partitions,p_id);
							VertexId i=begin_vid;
							while(i<end_vid){
								if(bitmap->get_bit(i)!=0&&(*((unsigned long*)(f_map+i))&mask)!=0){
									should_access_shard_col[partition_id]=true;
									
									should_access_shard_rc[p_id][partition_id]=true;
									break;
								}
								i++;
							}
						}

						//if(should_access_shard_col[partition_id]) break;
					}
				}

				#pragma omp barrier
				
				//free(vertex_l);
				
				int my_count1=0;
				int my_count2=0;
				long g_size=0;
				long h_size=0;
				for(int i=0;i<partitions;i++){
					if(should_access_shard[i]){
						my_count1++;
						for(int j=0;j<partitions;j++){
							g_size=g_size+fsize[i][j];
							if(should_access_shard_rc[i][j]){
								my_count2++;
								h_size=h_size+fsize[i][j];
							}
						}
					}
				}
					
				printf("GridGraph: %d  %ld  HPGraph: %d  %ld\n",my_count1*partitions,g_size,my_count2,h_size);
				
				munmap(f_map,vertices*sizeof(unsigned long));
				close(r);
			}
			
			
			//numa_free(vertex_l,vertices*sizeof(unsigned long));
/*			
			int my_count1=0;
            int my_count2=0;
            for(int ii=0;ii<partitions;ii++){
                if(should_access_shard[ii]){
                    my_count1++;
                }
                if(should_access_shard_col[ii]){
                    my_count2++;
                }
            }
			printf("This step access %d row_partitions and %d col_partitions\n",my_count1,my_count2);
*/
		}

		T value = zero;
		
		Queue<std::tuple<int, long, long> > tasks0(65536);
		Queue<std::tuple<int, long, long> > tasks1(65536);
		Queue<std::tuple<int, long, long> > tasks(65536);
		
		std::vector<std::thread> threads;
		long read_bytes = 0;
/*
		long total_bytes = 0;
		for (int i=0;i<partitions;i++) {
			if (!should_access_shard[i]) continue;
			for (int j=0;j<partitions;j++) {
				total_bytes += fsize[i][j];
			}
		}
		int read_mode;
		if (memory_bytes < total_bytes) {
			read_mode = O_RDONLY | O_DIRECT;
			// printf("use direct I/O\n");
		} else {
			read_mode = O_RDONLY;
			// printf("use buffered I/O\n");
		}
*/

		int fin1,fin2,fin;
		char* f_map1,*f_map2,*f_map;
		long offset1 = 0;
		long offset2 = 0;

		//cpu_set_t mask;
		struct stat st1;
		struct stat st2;
		struct stat st;

		switch(update_mode) {
		case 0: // source oriented update{
			{
				//double t1=get_time();
				
			long offset=0;
			fin=open((path+"/row").c_str(),O_RDONLY);
			fstat(fin,&st);
			posix_fadvise(fin, 0, 0, POSIX_FADV_SEQUENTIAL);
			f_map=(char*)mmap(NULL,st.st_size,PROT_READ,MAP_SHARED,fin,0);
			
			//double t2=get_time();
			//printf("mmap takes: %.2f\n",t2-t1);
			
			struct bitmask *bmask=numa_allocate_nodemask();
			numa_bitmask_setall(bmask);
			numa_tonodemask_memory(f_map,st.st_size,bmask);
			
			threads.clear();
			for (int ti=0;ti<parallelism;ti++) {
				threads.emplace_back([&](int thread_id){
					/*
					cpu_set_t mask;
					CPU_ZERO(&mask);
					CPU_SET(thread_id+1,&mask);
					sched_setaffinity(0,sizeof(mask),&mask);
					*/
					T local_value = zero;
					long local_read_bytes = 0;
					while (true) {
						int fin;
						long offset, length;
						std::tie(fin, offset, length) = tasks.pop();
						if (fin==-1) break;
						//char * buffer = buffer_pool[thread_id];
						//long bytes = pread(fin, buffer, length, offset);
						//assert(bytes>0);
						//if(offset>=st.st_size)
							//printf("offset: %lld file_size: %lld\n",offset,st.st_size);
						//printf("offset: %lld file_size: %lld length: %lld\n",offset,st.st_size,length);
						char* buffer=f_map+offset;
						//char* buffer=&f_map[offset];
						if(offset+length>st.st_size){
							length=st.st_size-offset;
							local_read_bytes+=length;
						}
						else
							local_read_bytes += length;
						// CHECK: start position should be offset % edge_unit
						for (long pos=offset % edge_unit;pos+edge_unit<=length;pos+=edge_unit) {
							Edge & e = *(Edge*)(buffer+pos);
					
							if (bitmap==nullptr || bitmap->get_bit(e.source)) {
								local_value += process(e);
							}
						}
					}
					write_add(&value, local_value);
					write_add(&read_bytes, local_read_bytes);
				}, ti);
			}
			//fin = open((path+"/row").c_str(), read_mode);
			
			//double t3=get_time();
			//printf("stream takes %.2f\n",t3-t2);
			
			for (int i=0;i<partitions;i++) {
				if (!should_access_shard[i]) continue;
				for (int j=0;j<partitions;j++) {
					long begin_offset = row_offset[i*partitions+j];
					if (begin_offset - offset >= PAGESIZE) {
						offset = begin_offset / PAGESIZE * PAGESIZE;
					}
					long end_offset = row_offset[i*partitions+j+1];
					if (end_offset <= offset) continue;
					while (end_offset - offset >= IOSIZE) {
						tasks.push(std::make_tuple(fin, offset, IOSIZE));
						offset += IOSIZE;
					}
					if (end_offset > offset) {
						tasks.push(std::make_tuple(fin, offset, (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE));
						offset += (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE;
					}
				}
			}
			
			//double t4=get_time();
			//printf("push takes %.2f\n",t4-t3);
			
			for (int i=0;i<parallelism;i++) {
				tasks.push(std::make_tuple(-1, 0, 0));
			}
			for (int i=0;i<parallelism;i++) {
				threads[i].join();
			}
			munmap(f_map,st.st_size);
			close(fin);
		}
			break;
		case 1: // target oriented update
			//fin = open((path+"/column").c_str(), read_mode);
			//posix_fadvise(fin, 0, 0, POSIX_FADV_SEQUENTIAL);
			{
			fin1=open((path+"/column-1").c_str(),O_RDONLY);
			fstat(fin1,&st1);
			posix_fadvise(fin1, 0, 0, POSIX_FADV_SEQUENTIAL);
			f_map1=(char*)mmap(NULL,st1.st_size,PROT_READ,MAP_SHARED,fin1,0);
			numa_tonode_memory(f_map1,st1.st_size,0);
			
			//printf("set to node 0\n");
			
			fin2=open((path+"/column-2").c_str(),O_RDONLY);
			fstat(fin2,&st2);
			posix_fadvise(fin2, 0, 0, POSIX_FADV_SEQUENTIAL);
			f_map2=(char*)mmap(NULL,st2.st_size,PROT_READ,MAP_SHARED,fin2,0);
			numa_tonode_memory(f_map2,st2.st_size,1);			

			//printf("set to node 1\n");
			
			for (int cur_partition=0;cur_partition<partitions;cur_partition+=partition_batch) {
				VertexId begin_vid, end_vid;
				begin_vid = get_partition_range(vertices, partitions, cur_partition).first;
				if (cur_partition+partition_batch>=partitions) {
					end_vid = vertices;
				} else {
					end_vid = get_partition_range(vertices, partitions, cur_partition+partition_batch).first;
				}
				pre_source_window(std::make_pair(begin_vid, end_vid));
				// printf("pre %d %d\n", begin_vid, end_vid);
				threads.clear();
				for (int ti=0;ti<parallelism;ti++) {
					threads.emplace_back([&](int thread_id){
					
						cpu_set_t mask;
						CPU_ZERO(&mask);
						CPU_SET(thread_id,&mask);
						sched_setaffinity(0,sizeof(mask),&mask);						

						T local_value = zero;
						long local_read_bytes = 0;
						while (true) {
							int fin;
							long offset, length;
							char* buffer;
							if((-1<thread_id&&thread_id<8)||(15<thread_id&&thread_id<24)){
								std::tie(fin, offset, length) = tasks0.pop();
								//if (fin==-1) break;
								if(fin==-1){
									std::tie(fin,offset,length)=tasks1.pop();
									if(fin==-1) break;
									
									buffer=f_map2+offset;
									if(offset+length>st2.st_size){
										length=st2.st_size-offset;
										local_read_bytes+=length;
									}
									else
										local_read_bytes+=length;
								}
								else{
									buffer=f_map1+offset;
								
									if(offset+length>st1.st_size){
										length=st1.st_size-offset;
										local_read_bytes+=length;
									}
									else
										local_read_bytes+=length;
								}
								
								//printf("thread_id: %d and in the first\n",thread_id);
							}
							else if((7<thread_id&&thread_id<16)||(23<thread_id&&thread_id<32)){
								std::tie(fin, offset, length) = tasks1.pop();
								//if (fin==-1) break;
								if(fin==-1){
									std::tie(fin, offset, length) = tasks0.pop();
									if(fin==-1) break;
									
									buffer=f_map1+offset;
								
									if(offset+length>st1.st_size){
										length=st1.st_size-offset;
										local_read_bytes+=length;
									}
									else
										local_read_bytes+=length;
								}
								else{
									buffer=f_map2+offset;
								
									if(offset+length>st2.st_size){
										length=st2.st_size-offset;
										local_read_bytes+=length;
									}
									else
										local_read_bytes+=length;
								}
								
								
								//printf("thread_id: %d and in the second\n",thread_id);
							}
							//std::tie(fin, offset, length) = tasks.pop();
							//if (fin==-1) break;
							//printf("offset: %lld file_size: %lld length: %lld\n",offset,st.st_size,length);
							//char* buffer=f_map+offset;
							//char * buffer = buffer_pool[thread_id];
							//printf("Thread_id %d\n",thread_id);
							//long bytes = pread(fin, buffer, length, offset);
							//assert(bytes>0);
							/*
							if(offset+length>st.st_size){
								length=st.st_size-offset;
								local_read_bytes+=length;
							}
							local_read_bytes += length;
							*/
							// CHECK: start position should be offset % edge_unit
							for (long pos=offset % edge_unit;pos+edge_unit<=length;pos+=edge_unit) {
								Edge & e = *(Edge*)(buffer+pos);
								if (bitmap==nullptr || bitmap->get_bit(e.source)) {
									local_value += process(e);
								}
							}
						}
						write_add(&value, local_value);
						write_add(&read_bytes, local_read_bytes);
					}, ti);
				}
				
				long offset = 0;
				if(partitions%2!=0){
					printf("partitions must %2=0\n");
					exit(0);
				}
				for (int j=0;j<record_col;j++) {
					
					//if(!should_access_shard_col[j]) continue;
					
					for (int i=cur_partition;i<cur_partition+partition_batch;i++) {
						if (i>=partitions) break;
						//if (!should_access_shard[i]) continue;
						
						if(!should_access_shard_rc[i][j]) continue;
	
						long begin_offset = column_offset1[j*partitions+i];
						if (begin_offset - offset >= PAGESIZE) {
							offset = begin_offset / PAGESIZE * PAGESIZE;
						}
						long end_offset = column_offset1[j*partitions+i+1];
						if (end_offset <= offset) continue;
						while (end_offset - offset >= IOSIZE) {
							tasks0.push(std::make_tuple(fin1, offset, IOSIZE));
							offset += IOSIZE;
						}
						if (end_offset > offset) {
							tasks0.push(std::make_tuple(fin1, offset, (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE));
							offset += (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE;
						}
					}
				}
				
				offset=0;
				for (int j=0;j<partitions-record_col;j++) {
					
					//if(!should_access_shard_col[j+partitions/2]) continue;
					
					for (int i=cur_partition;i<cur_partition+partition_batch;i++) {
						if (i>=partitions) break;
						//if (!should_access_shard[i]) continue;
						
						if(!should_access_shard_rc[i][j+record_col]) continue;
						
						long begin_offset = column_offset2[j*partitions+i];
						if (begin_offset - offset >= PAGESIZE) {
							offset = begin_offset / PAGESIZE * PAGESIZE;
						}
						long end_offset = column_offset2[j*partitions+i+1];
						if (end_offset <= offset) continue;
						while (end_offset - offset >= IOSIZE) {
							tasks1.push(std::make_tuple(fin2, offset, IOSIZE));
							offset += IOSIZE;
						}
						if (end_offset > offset) {
							tasks1.push(std::make_tuple(fin2, offset, (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE));
							offset += (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE;
						}
					}
				}
				
				//printf("task push finish!\n");
				
				for (int i=0;i<parallelism*100;i++) {
					tasks0.push(std::make_tuple(-1, 0, 0));
					tasks1.push(std::make_tuple(-1, 0, 0));
				}
				for (int i=0;i<parallelism;i++) {
					threads[i].join();
				}
				
				//printf("thread join finish!\n");
				
				post_source_window(std::make_pair(begin_vid, end_vid));
				// printf("post %d %d\n", begin_vid, end_vid);
			}
			munmap(f_map1,st1.st_size);
			munmap(f_map2,st2.st_size);
			close(fin1);
			close(fin2);
			}
			break;
		default:
			assert(false);
		}

		//munmap(f_map,st.st_size);
		//close(fin);
		// printf("streamed %ld bytes of edges\n", read_bytes);
		return value;
	}
};

#endif
