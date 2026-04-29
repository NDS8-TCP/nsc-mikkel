/* Matrix multiplication kernel 1 -- naive (global memory only)
 *
 * Each work-item computes one output element C[i,j].
 * It reads an entire row of A and an entire column of B from global memory.
 *
 *   A (N x N)        B (N x N)        C (N x N)
 *   [ . . . . ]      [ . c . . ]      [ . . . . ]
 *   [ r r r r ] x    [ . c . . ]  =   [ . * . . ]  <-- work-item (i,j)
 *   [ . . . . ]      [ . c . . ]      [ . . . . ]
 *
 *   reads N elements from row r of A  (global memory)
 *   reads N elements from col c of B  (global memory)
 *   writes one element C[i,j]         (global memory)
 *
 * Total global reads per output element: 2N
 * NDRange: (N, N) -- no local work-group needed
 */
__kernel void matmul(
    __global const float *a_device,
    __global const float *b_device,
    __global       float *result_device)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float result = 0;
    for(int k = 0; k < DIM; ++k) {
        result += a_device[i*DIM + k] * b_device[k*DIM+j];
    }

    result_device[i*DIM+j] = result;
}