/**
 * @brief Calculate the mean average precision (MAP) for a single query.
 *
 * Compile: gcc -shared -fPIC src/evaluation/map_calculator/MapCalculator.c -o src/evaluation/map_calculator/MapCalculator.so -fopenmp
 */
#include <omp.h>

double calculate_map(int actual_index, const int *rankings, int rankings_size)
{
    double precision_sum = 0.0;
    int num_relevant = 0;

    for (int i = 0; i < rankings_size; ++i)
    {
        if (rankings[i] == actual_index)
        {
            num_relevant++;
            precision_sum += (double)num_relevant / (i + 1);
        }
    }

    if (num_relevant == 0)
    {
        return 0.0;
    }

    return precision_sum / num_relevant;
}

double calculate_batch_map(const int *actual_indices,
                           const int *rankings_flat,
                           int num_queries,
                           int rankings_per_query)
{
    double map_sum = 0.0;

#pragma omp parallel
    {
        double local_sum = 0.0;

#pragma omp for schedule(static)
        for (int i = 0; i < num_queries; ++i)
        {
            const int *current_rankings = rankings_flat + (i * rankings_per_query);
            local_sum += calculate_map(actual_indices[i], current_rankings, rankings_per_query);
        }

#pragma omp atomic
        map_sum += local_sum;
    }

    return map_sum / num_queries;
}
