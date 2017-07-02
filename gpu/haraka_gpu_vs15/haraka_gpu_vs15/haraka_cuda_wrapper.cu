
#include "haraka_cuda_wrapper.h"

#include <iostream>
#include <cassert>

#ifdef _WIN32
	#include <Windows.h>
	#include <wincrypt.h>
#endif


#define checkCudaError(x) _checkCudaError(x, __FILE__, __LINE__)

inline void _checkCudaError(cudaError_t error, const char* file, const int line)
{
	if (error != cudaSuccess)
	{
		std::cout << "[" << file << ":" << line << "] got CUDA error " << error << ": " << cudaGetErrorString(error)
			<< std::endl;
		assert(false);
	}
}


cudaError_t harakaCuda512(const char* msgs, char* hashes, const uint32_t num_msgs)
{
	uint32_t msgs_per_stream = num_msgs / NUM_STREAMS;
	uint32_t remaining_msgs = num_msgs - NUM_STREAMS * msgs_per_stream;
	uint32_t msgs_to_alloc = msgs_per_stream ? msgs_per_stream : remaining_msgs;

	uint32_t grid_size = (msgs_to_alloc + MAX_THREAD - 1) / MAX_THREAD;
	dim3 block_dim(MAX_THREAD);

	checkCudaError(cudaSetDevice(0));

	char* msg_device;
	checkCudaError(cudaMalloc((void**)&msg_device, msgs_to_alloc * MSG_SIZE_BYTE_512 * sizeof(char)));

	char* hash_device;
	checkCudaError(cudaMalloc((void**)&hash_device, msgs_to_alloc * HASH_SIZE_BYTE * sizeof(char)));

	cudaStream_t streams[NUM_STREAMS];

	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaStreamCreate(&streams[i]);
		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[msgs_per_stream * MSG_SIZE_BYTE_512 * i], msgs_per_stream * MSG_SIZE_BYTE_512, cudaMemcpyHostToDevice, streams[i]));

		haraka512Kernel << <grid_size, block_dim, 0, streams[i] >> > ((uint64_t*)msg_device, (uint64_t*)hash_device, msgs_per_stream);

		checkCudaError(cudaMemcpyAsync(&hashes[msgs_per_stream * HASH_SIZE_BYTE * i], hash_device, msgs_per_stream * HASH_SIZE_BYTE, cudaMemcpyDeviceToHost, streams[i]));
	}

	if (remaining_msgs > 0)
	{
		cudaStream_t rem_stream;
		cudaStreamCreate(&rem_stream);
		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[msgs_per_stream * NUM_STREAMS * MSG_SIZE_BYTE_512], remaining_msgs * MSG_SIZE_BYTE_512, cudaMemcpyHostToDevice, rem_stream));

		haraka512Kernel << <grid_size, block_dim, 0, rem_stream >> > ((uint64_t*)msg_device, (uint64_t*)hash_device, remaining_msgs);

		checkCudaError(cudaMemcpyAsync(&hashes[msgs_per_stream * NUM_STREAMS * HASH_SIZE_BYTE], hash_device, remaining_msgs * HASH_SIZE_BYTE, cudaMemcpyDeviceToHost, rem_stream));
	}
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	cudaFree(msg_device);
	cudaFree(hash_device);

	return cudaSuccess;
}


cudaError_t harakaCuda256(const char* msgs, char* hashes, const uint32_t num_msgs)
{
	uint32_t msgs_per_stream = num_msgs / NUM_STREAMS;
	uint32_t remaining_msgs = num_msgs - NUM_STREAMS * msgs_per_stream;
	uint32_t msgs_to_alloc = msgs_per_stream ? msgs_per_stream : remaining_msgs;

	uint32_t grid_size = (msgs_to_alloc + MAX_THREAD - 1) / MAX_THREAD;
	dim3 block_dim(MAX_THREAD);

	checkCudaError(cudaSetDevice(0));

	char* msg_device;
	checkCudaError(cudaMalloc((void**)&msg_device, msgs_to_alloc * MSG_SIZE_BYTE_256 * sizeof(char)));

	char* hash_device;
	checkCudaError(cudaMalloc((void**)&hash_device, msgs_to_alloc * HASH_SIZE_BYTE * sizeof(char)));

	cudaStream_t streams[NUM_STREAMS];

	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaStreamCreate(&streams[i]);
		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[msgs_per_stream * MSG_SIZE_BYTE_256 * i], msgs_per_stream * MSG_SIZE_BYTE_256, cudaMemcpyHostToDevice, streams[i]));

		haraka256Kernel << <grid_size, block_dim, 0, streams[i] >> > ((uint64_t*)msg_device, (uint64_t*)hash_device, msgs_per_stream);

		checkCudaError(cudaMemcpyAsync(&hashes[msgs_per_stream * HASH_SIZE_BYTE * i], hash_device, msgs_per_stream * HASH_SIZE_BYTE, cudaMemcpyDeviceToHost, streams[i]));
	}

	if (remaining_msgs > 0)
	{
		cudaStream_t rem_stream;
		cudaStreamCreate(&rem_stream);
		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[msgs_per_stream * NUM_STREAMS * MSG_SIZE_BYTE_256], remaining_msgs * MSG_SIZE_BYTE_256, cudaMemcpyHostToDevice, rem_stream));

		haraka256Kernel << <grid_size, block_dim, 0, rem_stream >> > ((uint64_t*)msg_device, (uint64_t*)hash_device, remaining_msgs);

		checkCudaError(cudaMemcpyAsync(&hashes[msgs_per_stream * NUM_STREAMS * HASH_SIZE_BYTE], hash_device, remaining_msgs * HASH_SIZE_BYTE, cudaMemcpyDeviceToHost, rem_stream));
	}
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	cudaFree(msg_device);
	cudaFree(hash_device);

	return cudaSuccess;
}


#ifdef _WIN32
int harakaWinternitzCudaSign(const char* msgs, char* signatures, char* pub_keys, const uint32_t num_msgs)
{
	uint32_t msgs_per_stream = num_msgs / NUM_STREAMS;
	uint32_t remaining_msgs = num_msgs - NUM_STREAMS * msgs_per_stream;
	uint32_t msgs_to_alloc = msgs_per_stream ? msgs_per_stream : remaining_msgs;

	uint32_t grid_size_haraka = (msgs_to_alloc + MAX_THREAD - 1) / MAX_THREAD;
	uint32_t grid_size_winternitz = (T * msgs_to_alloc + MAX_THREAD - 1) / MAX_THREAD;
	dim3 block_dim(MAX_THREAD);

	cudaStream_t streams[NUM_STREAMS];

	HCRYPTPROV crypt_prov;
	if (!CryptAcquireContext(&crypt_prov, NULL, NULL, PROV_RSA_FULL, 0))
		return FAILED_TO_ACQUIRE_CRYPT_PROV;

	char* private_key = new char[T * HASH_SIZE_BYTE * msgs_to_alloc];

	char* msg_device;
	checkCudaError(cudaMalloc((void**)&msg_device, MSG_SIZE_BYTE_256 * msgs_to_alloc * sizeof(char)));

	char* b_device;
	checkCudaError(cudaMalloc((void**)&b_device, T * msgs_to_alloc * sizeof(char)));

	char* private_key_device;
	checkCudaError(cudaMalloc((void**)&private_key_device, T * HASH_SIZE_BYTE * msgs_to_alloc * sizeof(char)));

	char* public_key_device;
	checkCudaError(cudaMalloc((void**)&public_key_device, T * HASH_SIZE_BYTE * msgs_to_alloc * sizeof(char)));

	char* signature_device;
	checkCudaError(cudaMalloc((void**)&signature_device, T * HASH_SIZE_BYTE * msgs_to_alloc * sizeof(char)));

	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaStreamCreate(&streams[i]);

		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[MSG_SIZE_BYTE_256 * msgs_per_stream * i], MSG_SIZE_BYTE_256 * msgs_per_stream, cudaMemcpyHostToDevice, streams[i]));

		if (!CryptGenRandom(crypt_prov, T * HASH_SIZE_BYTE * msgs_per_stream, (BYTE*)(private_key)))
			return FAILED_TO_GENERATE_CRYPT_RAND_BYTES;

		checkCudaError(cudaDeviceSynchronize());
		harakaOTSKernel << <grid_size_haraka, block_dim, 0, streams[i] >> >((uint64_t*)msg_device, (uint16_t*)b_device, msgs_per_stream);

		checkCudaError(cudaMemcpyAsync((void *)private_key_device, private_key, T * HASH_SIZE_BYTE * msgs_per_stream, cudaMemcpyHostToDevice, streams[i]));

		harakaOTSCreateSignatureKernel << <grid_size_winternitz, block_dim, 0, streams[i] >> > ((uint64_t*)private_key_device, (uint64_t*)public_key_device, (uint64_t*)signature_device, (uint8_t*)b_device, T * msgs_per_stream);

		checkCudaError(cudaMemcpyAsync(&signatures[T * HASH_SIZE_BYTE * msgs_per_stream * i], signature_device, T * HASH_SIZE_BYTE * msgs_per_stream, cudaMemcpyDeviceToHost, streams[i]));

		checkCudaError(cudaMemcpyAsync(&pub_keys[T * HASH_SIZE_BYTE * msgs_per_stream * i], public_key_device, T * HASH_SIZE_BYTE * msgs_per_stream, cudaMemcpyDeviceToHost, streams[i]));
	}

	if (remaining_msgs > 0)
	{
		cudaStream_t rem_stream;
		cudaStreamCreate(&rem_stream);

		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[MSG_SIZE_BYTE_256 * msgs_per_stream * NUM_STREAMS], MSG_SIZE_BYTE_256 * remaining_msgs, cudaMemcpyHostToDevice, rem_stream));

		if (!CryptGenRandom(crypt_prov, T * HASH_SIZE_BYTE * remaining_msgs, (BYTE*)(private_key)))
			return FAILED_TO_GENERATE_CRYPT_RAND_BYTES;

		checkCudaError(cudaDeviceSynchronize());
		harakaOTSKernel << <grid_size_haraka, block_dim, 0, rem_stream >> >((uint64_t*)msg_device, (uint16_t*)b_device, remaining_msgs);

		checkCudaError(cudaMemcpyAsync((void *)private_key_device, private_key, T * HASH_SIZE_BYTE * remaining_msgs, cudaMemcpyHostToDevice, rem_stream));

		harakaOTSCreateSignatureKernel << <grid_size_winternitz, block_dim, 0, rem_stream >> > ((uint64_t*)private_key_device, (uint64_t*)public_key_device, (uint64_t*)signature_device, (uint8_t*)b_device, remaining_msgs * T);

		checkCudaError(cudaMemcpyAsync(&pub_keys[T * HASH_SIZE_BYTE * msgs_per_stream * NUM_STREAMS], public_key_device, T * HASH_SIZE_BYTE * remaining_msgs, cudaMemcpyDeviceToHost, rem_stream));

		checkCudaError(cudaMemcpyAsync(&signatures[T * HASH_SIZE_BYTE * msgs_per_stream * NUM_STREAMS], signature_device, T * HASH_SIZE_BYTE * remaining_msgs, cudaMemcpyDeviceToHost, rem_stream));
	}

	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	cudaFree(msg_device);
	cudaFree(b_device);
	cudaFree(private_key_device);
	cudaFree(public_key_device);
	cudaFree(signature_device);
	delete private_key;

	return SUCCESS;
}
#endif

int harakaWinternitzCudaSign(const char* msgs, const char* priv_keys, char* signatures, char* pub_keys, const uint32_t num_msgs)
{
	uint32_t msgs_per_stream = num_msgs / NUM_STREAMS;
	uint32_t remaining_msgs = num_msgs - NUM_STREAMS * msgs_per_stream;
	uint32_t msgs_to_alloc = msgs_per_stream ? msgs_per_stream : remaining_msgs;

	uint32_t grid_size_haraka = (msgs_to_alloc + MAX_THREAD - 1) / MAX_THREAD;
	uint32_t grid_size_winternitz = (T * msgs_to_alloc + MAX_THREAD - 1) / MAX_THREAD;
	dim3 block_dim(MAX_THREAD);

	cudaStream_t streams[NUM_STREAMS];

	char* msg_device;
	checkCudaError(cudaMalloc((void**)&msg_device, MSG_SIZE_BYTE_256 * msgs_to_alloc * sizeof(char)));

	char* b_device;
	checkCudaError(cudaMalloc((void**)&b_device, T * msgs_to_alloc * sizeof(char)));

	char* private_key_device;
	checkCudaError(cudaMalloc((void**)&private_key_device, T * HASH_SIZE_BYTE * msgs_to_alloc * sizeof(char)));

	char* public_key_device;
	checkCudaError(cudaMalloc((void**)&public_key_device, T * HASH_SIZE_BYTE * msgs_to_alloc * sizeof(char)));

	char* signature_device;
	checkCudaError(cudaMalloc((void**)&signature_device, T * HASH_SIZE_BYTE * msgs_to_alloc * sizeof(char)));

	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaStreamCreate(&streams[i]);

		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[MSG_SIZE_BYTE_256 * msgs_per_stream * i], MSG_SIZE_BYTE_256 * msgs_per_stream, cudaMemcpyHostToDevice, streams[i]));

		checkCudaError(cudaDeviceSynchronize());
		harakaOTSKernel << <grid_size_haraka, block_dim, 0, streams[i] >> >((uint64_t*)msg_device, (uint16_t*)b_device, msgs_per_stream);

		checkCudaError(cudaMemcpyAsync((void *)private_key_device, priv_keys, T * HASH_SIZE_BYTE * msgs_per_stream, cudaMemcpyHostToDevice, streams[i]));

		harakaOTSCreateSignatureKernel << <grid_size_winternitz, block_dim, 0, streams[i] >> > ((uint64_t*)private_key_device, (uint64_t*)public_key_device, (uint64_t*)signature_device, (uint8_t*)b_device, T * msgs_per_stream);

		checkCudaError(cudaMemcpyAsync(&signatures[T * HASH_SIZE_BYTE * msgs_per_stream * i], signature_device, T * HASH_SIZE_BYTE * msgs_per_stream, cudaMemcpyDeviceToHost, streams[i]));

		checkCudaError(cudaMemcpyAsync(&pub_keys[T * HASH_SIZE_BYTE * msgs_per_stream * i], public_key_device, T * HASH_SIZE_BYTE * msgs_per_stream, cudaMemcpyDeviceToHost, streams[i]));
	}

	if (remaining_msgs > 0)
	{
		cudaStream_t rem_stream;
		cudaStreamCreate(&rem_stream);

		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[MSG_SIZE_BYTE_256 * msgs_per_stream * NUM_STREAMS], MSG_SIZE_BYTE_256 * remaining_msgs, cudaMemcpyHostToDevice, rem_stream));

		checkCudaError(cudaDeviceSynchronize());
		harakaOTSKernel << <grid_size_haraka, block_dim, 0, rem_stream >> >((uint64_t*)msg_device, (uint16_t*)b_device, remaining_msgs);

		checkCudaError(cudaMemcpyAsync((void *)private_key_device, priv_keys, T * HASH_SIZE_BYTE * remaining_msgs, cudaMemcpyHostToDevice, rem_stream));

		harakaOTSCreateSignatureKernel << <grid_size_winternitz, block_dim, 0, rem_stream >> > ((uint64_t*)private_key_device, (uint64_t*)public_key_device, (uint64_t*)signature_device, (uint8_t*)b_device, remaining_msgs * T);

		checkCudaError(cudaMemcpyAsync(&pub_keys[T * HASH_SIZE_BYTE * msgs_per_stream * NUM_STREAMS], public_key_device, T * HASH_SIZE_BYTE * remaining_msgs, cudaMemcpyDeviceToHost, rem_stream));

		checkCudaError(cudaMemcpyAsync(&signatures[T * HASH_SIZE_BYTE * msgs_per_stream * NUM_STREAMS], signature_device, T * HASH_SIZE_BYTE * remaining_msgs, cudaMemcpyDeviceToHost, rem_stream));
	}

	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	cudaFree(msg_device);
	cudaFree(b_device);
	cudaFree(private_key_device);
	cudaFree(public_key_device);
	cudaFree(signature_device);

	return SUCCESS;
}


int harakaWinternitzCudaVerify(const char* msgs, const char* signatures, const char* pub_keys, const uint32_t num_msgs)
{
	uint32_t msgs_per_stream = num_msgs / NUM_STREAMS;
	uint32_t remaining_msgs = num_msgs - NUM_STREAMS * msgs_per_stream;
	uint32_t msgs_to_alloc = msgs_per_stream ? msgs_per_stream : remaining_msgs;

	uint32_t grid_size_haraka = (msgs_to_alloc + MAX_THREAD - 1) / MAX_THREAD;
	uint32_t grid_size_winternitz = (msgs_to_alloc * T + MAX_THREAD - 1) / MAX_THREAD;
	dim3 block_dim(MAX_THREAD);

	cudaStream_t streams[NUM_STREAMS];

	char* msg_device;
	checkCudaError(cudaMalloc((void**)&msg_device, MSG_SIZE_BYTE_256 * msgs_to_alloc * sizeof(char)));

	char* b_device;
	checkCudaError(cudaMalloc((void**)&b_device, T * msgs_to_alloc * sizeof(char)));

	char* signature_device;
	checkCudaError(cudaMalloc((void**)&signature_device, T * HASH_SIZE_BYTE * msgs_to_alloc * sizeof(char)));

	char* verification = new char[T * HASH_SIZE_BYTE * num_msgs];

	char* verification_device;
	checkCudaError(cudaMalloc((void**)&verification_device, T * HASH_SIZE_BYTE * msgs_to_alloc * sizeof(char)));

	for (int i = 0; i < NUM_STREAMS; ++i)
	{
		cudaStreamCreate(&streams[i]);

		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[MSG_SIZE_BYTE_256 * msgs_per_stream * i], msgs_per_stream * MSG_SIZE_BYTE_256, cudaMemcpyHostToDevice, streams[i]));

		harakaOTSKernel << <grid_size_haraka, block_dim, 0, streams[i] >> > ((uint64_t*)msg_device, (uint16_t*)b_device, msgs_per_stream);

		checkCudaError(cudaMemcpyAsync((void *)signature_device, &signatures[T * HASH_SIZE_BYTE * msgs_per_stream * i], T * HASH_SIZE_BYTE * msgs_per_stream, cudaMemcpyHostToDevice, streams[i]));

		harakaOTSCreateVerificationKernel << <grid_size_winternitz, block_dim, 0, streams[i] >> > ((uint64_t*)signature_device, (uint64_t*)verification_device, (uint8_t*)b_device, msgs_per_stream * T);

		checkCudaError(cudaMemcpyAsync(&verification[T * HASH_SIZE_BYTE * msgs_per_stream * i], verification_device, T * HASH_SIZE_BYTE * msgs_per_stream, cudaMemcpyDeviceToHost, streams[i]));
	}

	if (remaining_msgs > 0)
	{
		cudaStream_t rem_stream;
		cudaStreamCreate(&rem_stream);

		checkCudaError(cudaMemcpyAsync((void *)msg_device, &msgs[MSG_SIZE_BYTE_256 * msgs_per_stream * NUM_STREAMS], remaining_msgs * MSG_SIZE_BYTE_256, cudaMemcpyHostToDevice, rem_stream));

		harakaOTSKernel << <grid_size_haraka, block_dim, 0, rem_stream >> > ((uint64_t*)msg_device, (uint16_t*)b_device, remaining_msgs);

		checkCudaError(cudaMemcpyAsync((void *)signature_device, &signatures[T * HASH_SIZE_BYTE * msgs_per_stream * NUM_STREAMS], T * HASH_SIZE_BYTE * remaining_msgs, cudaMemcpyHostToDevice, rem_stream));

		harakaOTSCreateVerificationKernel << <grid_size_winternitz, block_dim, 0, rem_stream >> > ((uint64_t*)signature_device, (uint64_t*)verification_device, (uint8_t*)b_device, remaining_msgs * T);

		checkCudaError(cudaMemcpyAsync(&verification[T * HASH_SIZE_BYTE * msgs_per_stream * NUM_STREAMS], verification_device, T * HASH_SIZE_BYTE * remaining_msgs, cudaMemcpyDeviceToHost, rem_stream));
	}

	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	int verified = memcmp(pub_keys, verification, T * HASH_SIZE_BYTE * num_msgs) != 0;

	cudaFree(msg_device);
	cudaFree(b_device);
	cudaFree(signature_device);
	cudaFree(verification_device);
	delete verification;

	return verified;
}

int harakaBuildMerkleTree(char* tree, const uint32_t depth)
{
	uint32_t num_nodes = (1 << (depth + 1)) - 1;

	char* tree_device;
	checkCudaError(cudaMalloc((void**)&tree_device, HASH_SIZE_BYTE * num_nodes * sizeof(char)));

	checkCudaError(cudaMemcpyAsync((void *)&tree_device[((1 << depth) - 1) * HASH_SIZE_BYTE], &tree[((1 << depth) - 1) * HASH_SIZE_BYTE], (1 << depth) * HASH_SIZE_BYTE, cudaMemcpyHostToDevice));

	dim3 block_dim(MAX_THREAD);

	uint32_t rounds = depth / DEPTH_PER_KERNEL;
	uint32_t remaining_depth = depth;

	if (DEPTH_PER_KERNEL * rounds < depth)
	{
		uint32_t depth_first_kernel = depth - DEPTH_PER_KERNEL * rounds;
		uint32_t num_current_parents = (1 << (remaining_depth - 1));

		remaining_depth -= depth_first_kernel;

		uint32_t grid_size = (num_current_parents + MAX_THREAD - 1) / MAX_THREAD;

		harakaBuildMerkleTree << <grid_size, block_dim>> > ((uint64_t*)tree_device, num_current_parents, depth_first_kernel);
	}

	for (uint32_t i = 0; i < rounds; ++i)
	{
		uint32_t num_current_parents = (1 << (remaining_depth - 1));
		remaining_depth -= 6;

		uint32_t grid_size = (num_current_parents + MAX_THREAD - 1) / MAX_THREAD;

		harakaBuildMerkleTree << <grid_size, block_dim>> > ((uint64_t*)tree_device, num_current_parents, DEPTH_PER_KERNEL);
	}

	checkCudaError(cudaMemcpyAsync(&tree[0], (void *)&tree_device[0], ((1 << depth) - 1) * HASH_SIZE_BYTE, cudaMemcpyDeviceToHost));

	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());

	cudaFree(tree_device);

	return SUCCESS;
}