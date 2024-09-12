#!/bin/sh
clear
# 检测是否存在logs文件夹
if [ ! -d "logs" ]; then
  echo "logs 文件夹不存在，正在创建..."
  mkdir logs
else
  echo "logs 文件夹已存在。"
fi

cd cmake-build-release
cmake ..
cmake --build . --target example_search

# 4. 并行运行程序10次，并将每次的输出重定向到logs/1.log, logs/2.log, ..., logs/10.log
for i in $(seq 1 10); do
    echo "Running example_search: Iteration $i"
    ./example_search > ../logs/$i.log 2>&1 &
done

wait
echo "All done!"