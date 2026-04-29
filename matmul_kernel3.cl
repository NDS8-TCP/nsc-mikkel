/* Matrix multiplication kernel 3 -- tiled local memory
 *
 * A work-group of TILE_SIZE x TILE_SIZE work-items computes one
 * TILE_SIZE x TILE_SIZE block of C.  It sweeps in N/TILE_SIZE steps:
 * each step loads one tile of A (moving right) and one tile of B (moving
 * down) into __local memory, computes partial dot products, and accumulates
 * into a private register.  C is written exactly once at the end.
 *
 *  step 0 (tile_id=0)         step 1 (tile_id=1)         C (written once)
 *  A           B              A           B
 *  [# #|. .]   [# #|. .]     [. .|# #]   [. .|. .]      [* *|. .]
 *  [# #|. .]   [# #|. .]  +  [. .|# #]   [. .|. .]  +...= [* *|. .]
 *  [. .|. .]   [. .|. .]     [. .|. .]   [# #|. .]      [. .|. .]
 *  [. .|. .]   [. .|. .]     [. .|. .]   [# #|. .]      [. .|. .]
 *  ^->local     ^->local      ^->local     ^->local        ^fixed
 *
 *  A tile moves right, B tile moves down -- they always cover the same stripe.
 *  Each element read from global once, accessed TILE_SIZE times from __local.
 *  Global reads: 2*N^3/TILE_SIZE  (factor TILE_SIZE better than kernel 1).
 *
 * NDRange: (N, N) -- 2D, local work-group (TILE_SIZE, TILE_SIZE)
 */

#define NUM_TILES (DIM / TILE_SIZE)

__kernel void matmul3(
    __global const float *a_device,
    __global const float *b_device,
    __global       float *result_device)
{
    const int global_row = get_global_id(0);
    const int global_col = get_global_id(1);

    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);

    __local float a_work[TILE_SIZE][TILE_SIZE];
    __local float b_work[TILE_SIZE][TILE_SIZE];

    __private float result = 0.0f;
    for (int tile_id = 0; tile_id < NUM_TILES; ++tile_id) {
        const int tile_row_offset = TILE_SIZE * tile_id + local_row;
        const int tile_col_offset = TILE_SIZE * tile_id + local_col;

        a_work[local_col][local_row] = a_device[global_col * DIM + tile_row_offset];
        b_work[local_col][local_row] = b_device[tile_col_offset * DIM + global_row];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int iter = 0; iter < TILE_SIZE; ++iter) {
            result += a_work[local_col][iter] * b_work[iter][local_row];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result_device[global_col * DIM + global_row] = result;
}
