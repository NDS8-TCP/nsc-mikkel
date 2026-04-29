/* Matrix multiplication kernel 2 -- row-based restructuring
 *
 * Each work-item computes a full row of C.
 * Row i of A is read once and kept in the private accumulator register;
 * the inner loop sweeps all columns of B.
 *
 *   A (N x N)        B (N x N)        C (N x N)
 *   [ . . . . ]      [ all cols  ]    [ . . . . ]
 *   [ r r r r ] -->  [ ------->  ] =  [ * * * * ]  <-- work-item i
 *   [ . . . . ]      [           ]    [ . . . . ]
 *
 *   reads row r of A once into private register  (N global reads)
 *   reads all of B column by column              (N^2 global reads total per work-item)
 *   writes full row i of C                       (N global writes)
 *
 * No significant speedup over kernel 1 by itself, but the restructuring
 * is the prerequisite for tiling in kernel 3.
 * NDRange: (N,) -- 1D, one work-item per row
 */
__kernel void matmul2(
    __global const float *a_device,
    __global const float *b_device,
    __global       float *result_device)
{
    const int gid0 = get_global_id(0);

    const int row_offset = gid0 * DIM;

    for (int column_idx = 0; column_idx < DIM; ++column_idx) {
        float tmp_result = 0;
        int idx_b = column_idx;

        for (int itr = 0; itr < DIM; ++itr) {
            tmp_result += a_device[row_offset + itr] * b_device[idx_b];
            idx_b += DIM;
        }

        result_device[row_offset + column_idx] = tmp_result;
    }
}
