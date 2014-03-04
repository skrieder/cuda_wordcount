int main(){
  // read file to memory
  h_array = read("input.txt");
  
  // allocate device memory
  cudaMalloc(d_hashtable, sizeOfTable);
  cudaMalloc(d_array, sizeofInputFile);
  
  // move input array to device
  cudaMemcpy(d_array, h_array, size, hostToDevice);
  
  // first time through this is firstChunk
  cuda_wordcount<<<1, num_threads>>>(d_hashtable, d_array, num_threads);
  cudaDeviceSynchronize();
    
  // copy hash table back to host
  cudaMemcpy(h_hashtable, d_hashtable, size, deviceToHost);
  
  // iterate hash table
  iterate(h_hashtable);
  
  return 0;
}

__device__ void cuda_wordcount(hashtable d_hashtable, char **array, int num_threads){
  // parallel insert into table
  
}
