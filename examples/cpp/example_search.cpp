#include "../../hnswlib/hnswlib.h"
#include <string>
#include <fstream>
#include <vector>
#include <thread>

std::mutex file_mutex;

int solve(int m, std::fstream &fout)
{
    int dim = 100;             // Dimension of the elements
    int max_elements = 10000;  // Maximum number of elements, should be known beforehand
    int M = m;                 // Tightly connected with internal dimensionality of the data
                               // strongly affects the memory consumption
    int ef_construction = 100; // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(42);
    std::uniform_real_distribution<> distrib_real;
    float *data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++)
    {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++)
    {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    // count the time
    auto start = std::chrono::high_resolution_clock::now();
    float correct = 0;
    for (int i = 0; i < max_elements; i++)
    {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 10);
        // hnswlib::labeltype label = result.top().second;
        // if (label == i) correct++;
    }
    float recall = correct / max_elements;
    // std::cout << "Recall: " << recall << "\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::lock_guard<std::mutex> lock(file_mutex);

    std::cout << M << " " << elapsed_seconds.count() << "\n";

    // // Serialize index
    // std::string hnsw_path = "hnsw.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    // delete alg_hnsw;

    // // Deserialize index and check recall
    // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++)
    // {
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

int main()
{
    std::fstream fout("example_search.txt", std::ios::out);
    solve(4, fout);
    // std::vector<std::thread> threads;
    // for (int m = 2; m <= 100; m += 1)
    // {
    //     threads.push_back(std::thread(solve, m, std::ref(fout)));
    // }
    // for (auto &t : threads)
    // {
    //     t.join();
    // }
}