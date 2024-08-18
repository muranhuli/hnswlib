#include "../../hnswlib/hnswlib.h"
#include <H5Cpp.h>
#include <set>
#include "../../hnswlib/utils.h"

struct SolveResult
{
    double elapsed;
    float recall;
};

SolveResult solve(int M, int ef_construction)
{
    float disThreshold = 2e+06;
    size_t maxNum = 50;
    std::string filename = "/media/disk7T/liuyu/hdf5/fashion-mnist-784-euclidean.hdf5";

    hsize_t dims_out[2];
    auto data = DataRead::read_hdf5_float(filename, "/train",
                                          dims_out);
    int dim = int(dims_out[1]);
    int max_elements = int(dims_out[0]);

    // Initing index
    hnswlib::L2Space space(dim);
    auto *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, disThreshold,
                                                         maxNum, M,
                                                         ef_construction, 100, true);

    // Add data to index
    {
        for (int i = 0; i < max_elements; i++)
        {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data.get() + i * dim,
                                                                                                   20);
            std::set<int> result_label;
            while (!result.empty())
            {
                result_label.insert(int(result.top().second));
                result.pop();
            }

            if (result_label.empty() or
                !alg_hnsw->addPointToSuperNode(data.get() + i * dim, result_label))
            {
                hnswlib::labeltype label = alg_hnsw->cur_super_node_count;
                alg_hnsw->addPoint(data.get() + i * dim, label);
            }
        }
    }
    // alg_hnsw->hnsw_graph_info_stats();

    auto test_data = DataRead::read_hdf5_float(filename, "/test",
                                               dims_out);
    int test_max_elements = int(dims_out[0]);
    auto neighbor_data = DataRead::read_hdf5_int(filename,
                                                 "/neighbors", dims_out);
    int neighbor_max_elements = int(dims_out[0]);
    int neighbor_dim = int(dims_out[1]);

    // Query the elements for themselves and measure recall
    alg_hnsw->ef_ = 2 * ef_construction;
    alg_hnsw->ef_construction_ = 2 * ef_construction;
    float correct = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_max_elements; i++)
    {
        int k = 10;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(
                test_data.get() + i * dim, k);
        //  result提取出来，用于计算recall
        std::set<int> result_label;
        while (!result.empty())
        {
            result_label.insert(int(result.top().second));
            result.pop();
        }
        // 读取真实的neighbor
        std::set<int> neighbor;
        for (int j = 0; j < k; j++)
        {
            neighbor.insert(neighbor_data[i * neighbor_dim + j]);
        }
        // 计算recall, 集合交集/neighbor数
        std::set<int> intersection;
        std::set_intersection(result_label.begin(), result_label.end(), neighbor.begin(), neighbor.end(),
                              std::inserter(intersection, intersection.begin()));
        correct += static_cast<float>(intersection.size()) / static_cast<float>(neighbor.size());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(end - start).count();
    float recall = correct / test_max_elements;
    return SolveResult{test_max_elements * 1.0 / elapsed, recall};
}

int main()
{
    // int num_threads = 100;  // 最大线程数
    // omp_set_num_threads(num_threads);  // 设置 OpenMP 的最大线程数

    std::vector<int> M_values = {16, 32, 64};  // M 参数的取值
    std::vector<int> ef_values;  // ef_construction 参数的取值
    for (int ef = 5; ef <= 200; ef += 5)
    {
        ef_values.push_back(ef);
    }

    int num_count = M_values.size() * ef_values.size();  // 总共的任务数
    std::vector<SolveResult> results(num_count);
    std::vector<std::pair<int, int>> param_combinations(num_count); // 存储参数组合

    // 生成参数组合
    int idx = 0;
    for (int M: M_values)
    {
        for (int ef: ef_values)
        {
            param_combinations[idx] = {M, ef};
            idx++;
        }
    }

#pragma omp parallel for
    for (int i = 0; i < num_count; i++)
    {
        int M = param_combinations[i].first;
        int ef_construction = param_combinations[i].second;
        results[i] = solve(M, ef_construction);
    }

    // 输出结果到文件
    std::ofstream out_file("statistics_data.txt");
    for (int i = 0; i < num_count; i++)
    {
        out_file << results[i].elapsed << "/t" << results[i].recall << std::endl;
    }
    out_file.close();

    return 0;
}