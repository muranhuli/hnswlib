//
// Created by liuyu on 7/10/24.
//

#ifndef HNSWLIB_UTILS_H
#define HNSWLIB_UTILS_H

#include <chrono>
#include <string>
#include <iostream>
#include <memory>
#include <H5Cpp.h>
#include <vector>
#include <random>

class Time
{
private:
    std::string description; // 函数名
    std::string timeUnit; // 时间单位
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

public:
    explicit Time(const std::string& name, const std::string& unit = "s") :
        description(std::move(name)), timeUnit(std::move(unit))
    {
        start = std::chrono::high_resolution_clock::now();
    }

    ~Time()
    {
        end = std::chrono::high_resolution_clock::now();
        double elapsed = 0;

        if (timeUnit == "s")
        {
            elapsed = std::chrono::duration<double>(end - start).count();
        }
        else if (timeUnit == "ms")
        {
            elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        }
        else if (timeUnit == "us")
        {
            elapsed = std::chrono::duration<double, std::micro>(end - start).count();
        }
        else if (timeUnit == "ns")
        {
            elapsed = std::chrono::duration<double, std::nano>(end - start).count();
        }
        else
        {
            std::cerr << "Unsupported time unit: " << timeUnit << std::endl;
            return;
        }
        std::cout << description << "\tElapsed time:\t" << elapsed << " " << timeUnit << std::endl;
    }
};

class DataRead
{
public:
    static std::unique_ptr<float[]>
    read_hdf5_float(const std::string& filename, const std::string& dataset_name, hsize_t* dims_out)
    {
        const H5std_string& FILE_NAME(filename);
        const H5std_string& TRAIN_DATASET_NAME(dataset_name);
        H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(TRAIN_DATASET_NAME);
        H5::DataSpace dataspace = dataset.getSpace();

        // 输出数据的维度，个数
        // hsize_t dims_out[2];
        dataspace.getSimpleExtentDims(dims_out, NULL);
        int dim = static_cast<int>(dims_out[1]);
        int max_elements = static_cast<int>(dims_out[0]);

        std::unique_ptr<float[]> data(new float[dim * max_elements]);
        dataset.read(data.get(), H5::PredType::NATIVE_FLOAT, dataspace, dataspace);

        return data;
    }

    static std::unique_ptr<int[]>
    read_hdf5_int(const std::string& filename, const std::string& dataset_name, hsize_t* dims_out)
    {
        const H5std_string& FILE_NAME(filename);
        const H5std_string& TRAIN_DATASET_NAME(dataset_name);
        H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(TRAIN_DATASET_NAME);
        H5::DataSpace dataspace = dataset.getSpace();

        // 输出数据的维度，个数
        // hsize_t dims_out[2];
        dataspace.getSimpleExtentDims(dims_out, NULL);
        int dim = static_cast<int>(dims_out[1]);
        int max_elements = static_cast<int>(dims_out[0]);

        std::unique_ptr<int[]> data(new int[dim * max_elements]);
        dataset.read(data.get(), H5::PredType::NATIVE_INT, dataspace, dataspace);

        return data;
    }
};

void schedule(const std::string& content, int i, int max_elements)
{
    if (i == max_elements - 1)
        std::cout << "\r" << content << " " << i << "/" << max_elements << std::endl;
    else
        std::cout << "\r" << content << " " << i << "/" << max_elements << std::flush;
}

class CounterSingleton
{
public:
    static CounterSingleton& getInstance()
    {
        static CounterSingleton instance;
        return instance;
    }

    void increment()
    {
        count++;
    }

    // 清空计数器
    void clear()
    {
        count = 0;
    }

    int getCount() const
    {
        return count;
    }

private:
    int count;

    CounterSingleton() : count(0)
    {
    }

    CounterSingleton(const CounterSingleton&) = delete;
    CounterSingleton& operator=(const CounterSingleton&) = delete;
};


// 方法：https://blog.csdn.net/coffee_cream/article/details/109146143
class LSH
{
private:
    std::vector<float> r;
    float b;
    double w;
    size_t dimension;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

public:
    LSH(size_t dim, double width, unsigned seed = 42) : dimension(dim), w(width), distribution(0.0, 1.0),
                                                        generator(seed)
    {
        // 初始化随机向量r
        for (int i = 0; i < dimension; ++i)
        {
            r.push_back(distribution(generator));
        }
        // 随机选择偏移量b，范围在0到w之间
        b = (distribution(generator) * w) / 2.0;
    }

    int getHashValue(const void* v) const
    {
        double dotProduct = 0.0;
        for (int i = 0; i < dimension; ++i)
        {
            dotProduct += r[i] * ((float*)v)[i];
        }
        return static_cast<int>((dotProduct + b) / w);
    }
};


#endif //HNSWLIB_UTILS_H
