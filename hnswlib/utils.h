//
// Created by liuyu on 7/10/24.
//

#ifndef HNSWLIB_UTILS_H
#define HNSWLIB_UTILS_H

#include <chrono>
#include <string>
#include <iostream>

class Time
{
private:
    std::string description;  // 函数名
    std::string timeUnit;  // 时间单位
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

public:
    Time(const std::string &name, const std::string &unit = "s") :
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

#endif //HNSWLIB_UTILS_H
