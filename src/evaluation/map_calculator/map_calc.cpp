// map_calculator.cpp
#include <vector>
#include <numeric>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

/**
 * @brief Calculate the mean average precision (MAP) for a single query.
 * 
 * Compile: g++ -shared -fPIC src/evaluation/map_calculator/map_calculator.cpp -o src/evaluation/map_calculator/map_calculator.so
 */
extern "C"
{
    EXPORT double calculate_map(int actual_index, const int *rankings, int rankings_size)
    {
        double precision_sum = 0.0;
        int num_relevant = 0;

        for (int i = 0; i < rankings_size; ++i)
        {
            if (rankings[i] == actual_index)
            {
                num_relevant++;
                precision_sum += static_cast<double>(num_relevant) / (i + 1);
            }
        }

        if (num_relevant == 0)
        {
            return 0.0;
        }

        return precision_sum / num_relevant;
    }

    EXPORT double calculate_batch_map(const int *actual_indices,
                                      const int *rankings_flat,
                                      int num_queries,
                                      int rankings_per_query)
    {
        double map_sum = 0.0;

        for (int i = 0; i < num_queries; ++i)
        {
            // Calculate offset into flattened rankings array
            const int *current_rankings = rankings_flat + (i * rankings_per_query);
            map_sum += calculate_map(actual_indices[i], current_rankings, rankings_per_query);
        }

        return map_sum / num_queries;
    }
}