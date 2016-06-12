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
	bool * should_access_shard;
	bool * should_access_shard_col;
	long ** fsize;
	//char ** buffer_pool;
	char ** buffer_pool1;
	char ** buffer_pool2;
	//long * column_offset;
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
		/*
		buffer_pool = new char * [parallelism*1];
		for (int i=0;i<parallelism*1;i++) {
			buffer_pool[i] = (char *)memalign(PAGESIZE, IOSIZE);
			//buffer_pool[i] = (char*)numa_alloc_interleaved(IOSIZE);
			assert(buffer_pool[i]!=NULL);
			memset(buffer_pool[i], 0, IOSIZE);
		}
		*/
		
		buffer_pool1=new char* [parallelism/2];
		buffer_pool2=new char* [parallelism/2];
		for(int i=0;i<parallelism/2;i++){
			buffer_pool1[i] = (char*)numa_alloc_onnode(IOSIZE,0);
			buffer_pool2[i] = (char*)numa_alloc_onnode(IOSIZE,1);
			
			assert(buffer_pool1[i]!=NULL);
			memset(buffer_pool1[i], 0, IOSIZE);
			assert(buffer_pool2[i]!=NULL);
			memset(buffer_pool2[i], 0, IOSIZE);
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
		fscanf(fin_meta, "%d %d %ld %d", &edge_type, &vertices, &edges, &partitions);
		fclose(fin_meta);

		if (edge_type==0) {
			PAGESIZE = 4096;
		} else {
			PAGESIZE = 12288;
		}

		should_access_shard = new bool[partitions];
		should_access_shard_col = new bool[partitions];

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
/*
		column_offset = new long [partitions*partitions+1];
		int fin_column_offset = open((path+"/column_offset").c_str(), O_RDONLY);
		bytes = read(fin_column_offset, column_offset, sizeof(long)*(partitions*partitions+1));
		assert(bytes==sizeof(long)*(partitions*partitions+1));
		close(fin_column_offset);
*/
		column_offset1 = new long [partitions*partitions/2+1];
		int fin_column_offset1 = open((path+"/column_offset-1").c_str(), O_RDONLY);
		bytes = read(fin_column_offset1, column_offset1, sizeof(long)*(partitions*partitions/2+1));
		assert(bytes==sizeof(long)*(partitions*partitions/2+1));
		close(fin_column_offset1);
		
		column_offset2 = new long [partitions*partitions/2+1];
		int fin_column_offset2 = open((path+"/column_offset-2").c_str(), O_RDONLY);
		bytes = read(fin_column_offset2, column_offset2, sizeof(long)*(partitions*partitions/2+1));
		assert(bytes==sizeof(long)*(partitions*partitions/2+1));
		close(fin_column_offset2);
		
		row_offset = new long [partitions*partitions+1];
		int fin_row_offset = open((path+"/row_offset").c_str(), O_RDONLY);
		bytes = read(fin_row_offset, row_offset, sizeof(long)*(partitions*partitions+1));
		assert(bytes==sizeof(long)*(partitions*partitions+1));
		close(fin_row_offset);

		//printf("row_offset0: %ld row_offset_end: %ld and col_offset0: %ld col_offset_end: %ld\n",row_offset[0],row_offset[partitions*partitions],column_offset[0],column_offset[partitions*partitions]);
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
		int x = (int)ceil(bytes / (0.8 * memory_bytes));
		partition_batch = partitions / x;
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
		
		if(col_del==0){
			//double t1=get_time();
			if (bitmap==nullptr) {
				for (int i=0;i<partitions;i++) {
					should_access_shard[i] = true;
					should_access_shard_col[i] = true;
				}
			} else {
				for (int i=0;i<partitions;i++) {
					should_access_shard[i] = false;
					should_access_shard_col[i] = true;
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
			}
			
			//printf("no select takes %.2f\n",get_time()-t1);
		}
		else if(col_del==1){
			
			
			//printf("malloc takes %.2f\n",get_time()-t2);
			if (bitmap==nullptr) {
				for (int i=0;i<partitions;i++) {
					should_access_shard[i] = true;
					should_access_shard_col[i] = true;
				}
			} else {
				
				//double t2=get_time();
				int r=open((path+"/record_c").c_str(),O_RDONLY|O_DIRECT);
				posix_fadvise(r, 0, 0, POSIX_FADV_SEQUENTIAL);
				//unsigned long* r_buffer=(unsigned long*)numa_alloc_interleaved(256*1024*1024);
/*       		unsigned long *vertex_l=(unsigned long*)malloc(vertices*sizeof(unsigned long));
        	unsigned long count=read(r,vertex_l,sizeof(unsigned long)*vertices);
			if(count!=vertices*sizeof(unsigned long)){
				printf("read count: %d  r: %d\n",count,r);
				exit(0);
			}
	        close(r);
*/			//t3=get_time();
				//printf("open success!\n");
				unsigned long * f_map=(unsigned long *)mmap(NULL,vertices*sizeof(unsigned long),PROT_READ,MAP_SHARED,r,0);
				
				//printf("mmap success!\n");
				//struct bitmask *bmask=numa_allocate_nodemask();
				//numa_bitmask_setall(bmask);
				//numa_tonodemask_memory(f_map,vertices*sizeof(unsigned long),bmask);
				//numa_free_nodemask(bmask);
				
			
				for (int i=0;i<partitions;i++) {
					should_access_shard[i] = false;
					should_access_shard_col[i] = false;
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
				
				//printf("opm1 finish!\n");
				
				#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
				for(int partition_id=0;partition_id<partitions;partition_id++){
					unsigned long mask=0;
					__sync_fetch_and_or(&mask,1<<partition_id);
					for(int p_id=0;p_id<partitions;p_id++){
						if(should_access_shard[p_id]){						
							VertexId begin_vid,end_vid;
							std::tie(begin_vid,end_vid)=get_partition_range(vertices,partitions,p_id);
							
							//if(end_vid>=vertices){
								//printf("end_vid: %d and vertices: %d\n",end_vid,vertices);
							//}
							//lseek(r,begin_vid*sizeof(unsigned long),SEEK_SET);
							//printf("end_vid-begin_vid: %d\n",end_vid-begin_vid);
							//unsigned long count=read(r,r_buffer,sizeof(unsigned long)*(end_vid-begin_vid));
							//printf("end_vid-begin_vid: %d\n",end_vid-begin_vid);
							//if(count!=)
							
							unsigned long i=begin_vid;
							while(i<end_vid){
								if(bitmap->get_bit(i)!=0&&(*((unsigned long*)(f_map+i))&mask)!=0){
									should_access_shard_col[partition_id]=true;
									break;
								}
								i++;
							}
						}

						if(should_access_shard_col[partition_id]) break;
					}
				}

				#pragma omp barrier	
				
				//printf("omp2 finish!\n");
				
				int ret=munmap(f_map,vertices*sizeof(unsigned long));
				assert(ret==0);
				//numa_free(r_buffer,256*1024*1024);
				close(r);
				//printf("munmap success!\n");
			}
			
			//free(vertex_l);
			
			//printf("select takes %.2f\n",get_time()-t2);
		}

		T value = zero;
		Queue<std::tuple<int, long, long> > tasks(65536);
		Queue<std::tuple<int, long, long> > tasks0(65536);
		Queue<std::tuple<int, long, long> > tasks1(65536);
		std::vector<std::thread> threads;
		long read_bytes = 0;

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

		int fin;
		long offset = 0;
		int fin1,fin2;
		long offset1,offset2;
		
		switch(update_mode) {
		case 0: // source oriented update
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
						std::tie(fin, offset, length) = tasks.pop();
						if (fin==-1) break;
						//char * buffer = buffer_pool[thread_id];
						
						//long bytes = pread(fin, buffer, length, offset);
						char* buffer;
						long bytes;
						if(thread_id>-1&&thread_id<8){
							buffer=buffer_pool1[thread_id];
							bytes=pread(fin,buffer,length,offset);
						}
						else if(thread_id>15&&thread_id<24){
							buffer=buffer_pool1[thread_id-8];
							bytes=pread(fin,buffer,length,offset);
						}
						else if(thread_id>7&&thread_id<16){
							buffer=buffer_pool2[thread_id-8];
							bytes=pread(fin,buffer,length,offset);
						}
						else{
							buffer=buffer_pool2[thread_id-16];
							bytes=pread(fin,buffer,length,offset);
						}
						assert(bytes>0);
						local_read_bytes += bytes;
						// CHECK: start position should be offset % edge_unit
						for (long pos=offset % edge_unit;pos+edge_unit<=bytes;pos+=edge_unit) {
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
			fin = open((path+"/row").c_str(), read_mode);
			posix_fadvise(fin, 0, 0, POSIX_FADV_SEQUENTIAL);
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
			for (int i=0;i<parallelism;i++) {
				tasks.push(std::make_tuple(-1, 0, 0));
			}
			for (int i=0;i<parallelism;i++) {
				threads[i].join();
			}
			close(fin);
			break;
		case 1: // target oriented update
			//fin = open((path+"/column").c_str(), read_mode);
			//posix_fadvise(fin, 0, 0, POSIX_FADV_SEQUENTIAL);
			
			fin1 = open((path+"/column-1").c_str(), read_mode);
			posix_fadvise(fin1, 0, 0, POSIX_FADV_SEQUENTIAL);
			
			fin2 = open((path+"/column-2").c_str(), read_mode);
			posix_fadvise(fin2, 0, 0, POSIX_FADV_SEQUENTIAL);

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
							//std::tie(fin, offset, length) = tasks.pop();
							//if (fin==-1) break;
							//char * buffer = buffer_pool[thread_id];
							//long bytes = pread(fin, buffer, length, offset);
							//printf("bytes: %d and length: %d\n",bytes,length);
							
							char* buffer;
							long bytes;
							if(thread_id>-1&&thread_id<8){
								std::tie(fin, offset, length) = tasks0.pop();
								//if (fin==-1) break;
								if(fin==-1){
									std::tie(fin,offset,length)=tasks1.pop();
									if(fin==-1) break;
								}
								buffer=buffer_pool1[thread_id];
								bytes=pread(fin,buffer,length,offset);
							}
							else if(thread_id>15&&thread_id<24){
								std::tie(fin, offset, length) = tasks0.pop();
								//if (fin==-1) break;
								if(fin==-1){
									std::tie(fin,offset,length)=tasks1.pop();
									if(fin==-1) break;
								}
								buffer=buffer_pool1[thread_id-8];
								bytes=pread(fin,buffer,length,offset);
							}
							else if(thread_id>7&&thread_id<16){
								std::tie(fin, offset, length) = tasks1.pop();
								//if (fin==-1) break;
								if(fin==-1){
									std::tie(fin,offset,length)=tasks0.pop();
									if(fin==-1) break;
								}
								buffer=buffer_pool2[thread_id-8];
								bytes=pread(fin,buffer,length,offset);
							}
							else{
								std::tie(fin, offset, length) = tasks1.pop();
								//if (fin==-1) break;
								if(fin==-1){
									std::tie(fin,offset,length)=tasks0.pop();
									if(fin==-1) break;
								}
								buffer=buffer_pool2[thread_id-16];
								bytes=pread(fin,buffer,length,offset);
							}
							
							assert(bytes>0);
							local_read_bytes += bytes;
							// CHECK: start position should be offset % edge_unit
							for (long pos=offset % edge_unit;pos+edge_unit<=bytes;pos+=edge_unit) {
								Edge & e = *(Edge*)(buffer+pos);
								if (e.source < begin_vid || e.source >= end_vid) {
									continue;
								}
								if (bitmap==nullptr || bitmap->get_bit(e.source)) {
									local_value += process(e);
								}
							}
						}
						write_add(&value, local_value);
						write_add(&read_bytes, local_read_bytes);
					}, ti);
				}
				offset = 0;
				for (int j=0;j<partitions/2;j++) {
					
					if(!should_access_shard_col[j]) continue;
					
					for (int i=cur_partition;i<cur_partition+partition_batch;i++) {
						if (i>=partitions) break;
						if (!should_access_shard[i]) continue;
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
				
				offset = 0;
				for (int j=0;j<partitions/2;j++) {
					
					if(!should_access_shard_col[j+partitions/2]) continue;
					
					for (int i=cur_partition;i<cur_partition+partition_batch;i++) {
						if (i>=partitions) break;
						if (!should_access_shard[i]) continue;
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
				for (int i=0;i<parallelism*100;i++) {
					tasks0.push(std::make_tuple(-1, 0, 0));
					tasks1.push(std::make_tuple(-1, 0, 0));
				}
				for (int i=0;i<parallelism;i++) {
					threads[i].join();
				}
				post_source_window(std::make_pair(begin_vid, end_vid));
				// printf("post %d %d\n", begin_vid, end_vid);
			}
			
			close(fin1);
			close(fin2);

			break;
		default:
			assert(false);
		}

		//close(fin);
		// printf("streamed %ld bytes of edges\n", read_bytes);
		return value;
	}
};

#endif
