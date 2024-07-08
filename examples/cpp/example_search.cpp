#include "../../hnswlib/hnswlib.h"
#include <chrono>

int main()
{
    int dim =16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 32;                 // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    float disThreshold = 20;
    int maxNum = 200;

    std::cout<<"dim="<<dim<<" max_elements="<<max_elements<<" M="<<M<<" ef_construction="<<ef_construction<<std::endl;
    std::cout<<"disThreshold="<<disThreshold<<" maxNum="<<maxNum<<std::endl;

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M,
                                                                                    ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(12);
    std::uniform_real_distribution<> distrib_real;
    float *data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++)
    {
        data[i] = distrib_real(rng);
    }


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
    std::cout<<std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "构图时间: " << duration.count() << "秒" << std::endl;

    // std::cout<<alg_hnsw->node2SuperNode[0]<<std::endl;

    // supernode size
    // std::cout << "The size of supernode is " << alg_hnsw->super_node_set_.size() << std::endl;
    alg_hnsw->calculateSpaceCost();

    start = std::chrono::high_resolution_clock::now();
    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++)
    {
        // std::cout<<"\rann "<<i<<"/"<<max_elements;
        // std::cout.flush();
        if (i==12)
            int a =1;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
        // else {
        //     std::cout<<"error label = "<<label<<std::endl;
        //     std::cout << "true label = " << i << std::endl;
        //     std::cout<<"dist = "<<result.top().first<<std::endl;
        // }
    }
    std::cout<<std::endl;
    float recall = correct / max_elements;
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