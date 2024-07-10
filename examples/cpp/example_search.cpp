#include "../../hnswlib/hnswlib.h"
#include <H5Cpp.h>
#include <set>
#include "../../hnswlib/utils.h"


int main()
{
    Time t("All Time");
    int M = 16;                 // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // 读取数据
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

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M,
                                                                                    ef_construction);

    // Add data to index
    for (int i = 0; i < max_elements; i++)
    {
        alg_hnsw->addPoint(data + i * dim, i);
    }

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

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < test_max_elements; i++)
    {
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
        correct += static_cast<float>(intersection.size()) / static_cast<float>(neighbor.size());
    }
    float recall = correct / test_max_elements;
    std::cout << "Recall: " << recall << "\n";

    // // Serialize index
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

    delete[] data;
    delete alg_hnsw;
    return 0;
}
