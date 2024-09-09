#include "../../hnswlib/hnswlib.h"
#include <H5Cpp.h>
#include <set>
#include "../../hnswlib/utils.h"

int main()
{
    int M = 64;                 // Tightly connected with internal dimensionality of the data
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    float disThreshold = 2e+06;
    size_t maxNum = 50;
    std::string filename = "/data/liuyu/hdf5/fashion-mnist-784-euclidean.hdf5";

    hsize_t dims_out[2];
    auto data = DataRead::read_hdf5_float(filename, "/train",
                                          dims_out);
    int dim = int(dims_out[1]);
    int max_elements = int(dims_out[0]);

    std::cout << "dim=" << dim << " max_elements=" << max_elements << " M=" << M << " ef_construction="
              << ef_construction << std::endl;
    std::cout << "disThreshold=" << disThreshold << " maxNum=" << maxNum << std::endl;

    // Initing index
    hnswlib::L2Space space(dim);
    auto *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, disThreshold,
                                                         maxNum, M,
                                                         ef_construction, 100, true);

    // Add data to index
    {
        Time time("Build Index");
        for (int i = 0; i < max_elements; i++)
        {
            schedule("AddPoint", i, max_elements);
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data.get() + i * dim, 20);
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
    alg_hnsw->hnsw_graph_info_stats();

    auto test_data = DataRead::read_hdf5_float(filename, "/test",
                                               dims_out);
    int test_max_elements = int(dims_out[0]);
    auto neighbor_data = DataRead::read_hdf5_int(filename,
                                                 "/neighbors", dims_out);
    int neighbor_max_elements = int(dims_out[0]);
    int neighbor_dim = int(dims_out[1]);

    // Query the elements for themselves and measure recall
    alg_hnsw->ef_construction_ = 400;
    alg_hnsw->ef_ = 400;
    alg_hnsw->num1 = 0;
    alg_hnsw->num2 = 0;
    float correct = 0;
    {
        Time time("KNN Search");
        std::ofstream fout("calculDisNum.txt");
        for (int i = 0; i < test_max_elements; i++)
        {
            schedule("ANN", i, test_max_elements);
            int k = 10;
            CounterSingleton::getInstance().clear();
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
            fout << CounterSingleton::getInstance().getCount() << std::endl;
        }
        fout.close();
    }
    float recall = correct / test_max_elements;
    std::cout << "Recall: " << recall << "\n";
    std::cout<<"Num: "<<alg_hnsw->num2<<"/"<<alg_hnsw->num1<<std::endl;

    // Serialize index
    // std::string hnsw_path = "hnsw.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    // delete alg_hnsw;
    //
    // // Deserialize index and check recall
    // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // recall = (float)correct / max_elements;
    // std::cout << "Recall of deserialized index: " << recall << "\n";
    delete alg_hnsw;
    return 0;
}