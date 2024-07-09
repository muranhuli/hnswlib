#include "../../hnswlib/hnswlib.h"
#include <chrono>
#include <H5Cpp.h>
#include <set>

int main(int argc, char **argv)
{

    int M = std::stoi(argv[1]);                 // Tightly connected with internal dimensionality of the data
    int ef_construction = std::stoi(argv[2]);  // Controls index search speed/build speed tradeoff
    float disThreshold = std::stoi(argv[3]) ;
    int maxNum = std::stoi(argv[4]);

    const H5std_string FILE_NAME("/media/disk7T/liuyu/hdf5/fashion-mnist-784-euclidean.hdf5");
    const H5std_string TRAIN_DATASET_NAME("/train");
    H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(TRAIN_DATASET_NAME);
    H5::DataSpace dataspace = dataset.getSpace();
    // 输出数据的维度，个数
    hsize_t dims_out[2];
    int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
    int dim = int(dims_out[1]);
    int max_elements = int(dims_out[0]);
    auto *data = new float[dim * max_elements];
    dataset.read(data, H5::PredType::NATIVE_FLOAT, dataspace, dataspace);

    std::cout << "dim=" << dim << " max_elements=" << max_elements << " M=" << M << " ef_construction="
              << ef_construction << std::endl;
    std::cout << "disThreshold=" << disThreshold << " maxNum=" << maxNum << std::endl;

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M,
                                                                                    ef_construction);

    // Add data to index
    // 统计程序运行时间
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_elements; i++)
    {
        // std::cout<<"\raddPoint "<<i<<"/"<<max_elements;
        // std::cout.flush();
        // 首先找最小点
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        if (result.empty() or
            !alg_hnsw->addPointToSuperNode(data + i * dim, alg_hnsw->node2SuperNode[result.top().second],
                                           disThreshold, maxNum))
            alg_hnsw->addPoint(data + i * dim, i);
    }
    std::cout << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "构图时间: " << duration.count() << "秒" << std::endl;


    alg_hnsw->calculateSpaceCost();

    // 读取test数据
    const H5std_string TEST_DATASET_NAME("/test");
    dataset = file.openDataSet(TEST_DATASET_NAME);
    dataspace = dataset.getSpace();
    // 输出数据的维度，个数
    ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
    int test_max_elements = int(dims_out[0]);
    auto *test_data = new float[dim * test_max_elements];
    dataset.read(test_data, H5::PredType::NATIVE_FLOAT, dataspace, dataspace);

    // 读取查询结果top-k，neighbor
    const H5std_string NEIGHBOR_DATASET_NAME("/neighbors");
    dataset = file.openDataSet(NEIGHBOR_DATASET_NAME);
    dataspace = dataset.getSpace();
    // 输出数据的维度，个数
    ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
    int neighbor_max_elements = int(dims_out[0]);
    int neighbor_dim = int(dims_out[1]);
    auto *neighbor_data = new int[neighbor_dim * neighbor_max_elements];
    dataset.read(neighbor_data, H5::PredType::NATIVE_INT, dataspace, dataspace);

    start = std::chrono::high_resolution_clock::now();
    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < test_max_elements; i++)
    {
        // std::cout<<"\rann "<<i<<"/"<<max_elements;
        // std::cout.flush();
        int k = 10;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(test_data + i * dim, k);
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
        float tmp_recall = static_cast<float>(intersection.size()) / static_cast<float>(neighbor.size());
        correct += tmp_recall;
    }
    std::cout << std::endl;
    float recall = correct / test_max_elements;
    std::cout << "Recall: " << recall << "\n";
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "查询时间: " << duration.count() << "秒" << std::endl;

    // Serialize index
    // std::string hnsw_path = "hnsw.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    // delete alg_hnsw;

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

    delete[] data;
    delete alg_hnsw;
    return 0;
}