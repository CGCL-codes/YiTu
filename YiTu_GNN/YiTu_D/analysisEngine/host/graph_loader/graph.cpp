#include "graph.h"
#include <cstdlib>
#include <iostream>
#include <fstream>

void Graph::loadFile(
    const std::string& gName,
    std::vector<std::vector<int>> &data
)
{
    std::ifstream fhandle(gName.c_str());
    if (!fhandle.is_open()) {
        HERE;
        std::cout << "Failed to open " << gName << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    while (std::getline(fhandle, line)) {
        std::istringstream iss(line);
        data.push_back(
            std::vector<int>(std::istream_iterator<int>(iss),
                             std::istream_iterator<int>())
        );
    }
    fhandle.close();

    std::cout << "Graph " << gName << " is loaded." << std::endl;
}

//获取图中最大顶点ID
int Graph::getMaxIdx(const std::vector<std::vector<int>> &data) {
    int maxIdx = data[0][0];
    for (auto it1 = data.begin(); it1 != data.end(); it1++) {
        for (auto it2 = it1->begin(); it2 != it1->end(); it2++) {
            if (maxIdx <= (*it2)) {
                maxIdx = *it2;
            }
        }
    }
    return maxIdx;
}

//获取图中最小顶点ID
int Graph::getMinIdx(const std::vector<std::vector<int>> &data) {
    int minIdx = data[0][0];
    for (auto it1 = data.begin(); it1 != data.end(); it1++) {
        for (auto it2 = it1->begin(); it2 != it1->end(); it2++) {
            if (minIdx >= (*it2)) {
                minIdx = *it2;
            }
        }
    }
    return minIdx;
}

//构造函数
Graph::Graph(const std::string& gName) {

    // Check if it is undirectional graph
    auto found = gName.find("ungraph", 0);
    if (found != std::string::npos)
        isUgraph = true;
    else
        isUgraph = false;

    std::vector<std::vector<int>> data;
    loadFile(gName, data);
    vertexNum = getMaxIdx(data) + 1;
    edgeNum = (int)data.size();
    if (isUgraph) edgeNum = edgeNum * 2;
    std::cout << "vertex num: " << vertexNum << std::endl;
    std::cout << "edge num: " << edgeNum << std::endl;

    for (int i = 0; i < vertexNum; i++) {
        Vertex* v = new Vertex(i);
        vertices.push_back(v);
    }

    for (auto it = data.begin(); it != data.end(); it++) {
        int srcIdx = (*it)[0];
        int dstIdx = (*it)[1];
        vertices[srcIdx]->outVid.push_back(dstIdx);
        vertices[dstIdx]->inVid.push_back(srcIdx);
        if (isUgraph && srcIdx != dstIdx) {
            vertices[dstIdx]->outVid.push_back(srcIdx);
            vertices[srcIdx]->inVid.push_back(dstIdx);
        }
    }

    for (auto it = vertices.begin(); it != vertices.end(); it++) {
        (*it)->inDeg = (int)(*it)->inVid.size();
        (*it)->outDeg = (int)(*it)->outVid.size();
    }
}

CSR::CSR(const Graph &g) : vertexNum(g.vertexNum), edgeNum(g.edgeNum) {
    rpao.resize(vertexNum + 1);
    rpai.resize(vertexNum + 1);
    rpao[0] = 0;
    rpai[0] = 0;
    for (int i = 0; i < vertexNum; i++) {
        rpao[i + 1] = rpao[i] + g.vertices[i]->outDeg;
        rpai[i + 1] = rpai[i] + g.vertices[i]->inDeg;
    }

    // sort the input and output vertex
    for (int i = 0; i < vertexNum; i++) {
        //std::sort(g.vertices[i]->outVid.begin(), g.vertices[i]->outVid.end());
        //std::sort(g.vertices[i]->inVid.begin(), g.vertices[i]->inVid.end());
        for (auto id : g.vertices[i]->outVid) {
            ciao.push_back(id);
        }
        for (auto id : g.vertices[i]->inVid) {
            ciai.push_back(id);
        }
    }
#if 0
    for (int i = 0; i < edgeNum; i++) {
        eProps.push_back(rand() % 10);
    }
#endif
}

//将CSR格式图数据保存到文件
int CSR::save2File(const std::string &fName)
{
    std::ostringstream command;
    command << "mkdir -p ";
    command <<"csr/";
    command << fName;
    int ret = system(command.str().c_str());
    if (ret < 0)
    {
        return ret;
    }
    std::ostringstream offsetsName, columnsName;
    offsetsName << "csr/";
    offsetsName << fName;
    offsetsName << "/csr_offsets.txt";
    std::ofstream offset;
    offset.open(offsetsName.str().c_str());

    columnsName << "csr/";
    columnsName << fName;
    columnsName << "/csr_columns.txt";
    std::ofstream columns;
    columns.open(columnsName.str().c_str());

    offset << (rpao.size() - 1) << std::endl;
    for (auto item : rpao)
    {
        offset << item << std::endl;
    }
    offset.flush();
    offset.close();

    columns << ciao.size() << std::endl;
    for (auto item : ciao)
    {
        columns << item << std::endl;
    }
    columns.flush();
    columns.close();

    return 0;


}

CSR_BLOCK::CSR_BLOCK(
    const int _cordx,
    const int _cordy,
    CSR* csr
) : cordx(_cordx), cordy(_cordy)
{
    vertexNum = PARTITION_SIZE;
    srcStart = cordx * PARTITION_SIZE;
    srcEnd = (cordx + 1) * PARTITION_SIZE;
    if (srcEnd > csr->vertexNum) {
        srcEnd = csr->vertexNum;
    }
    vertexNum = srcEnd - srcStart;

    sinkStart = cordy * PARTITION_SIZE;
    sinkEnd = (cordy + 1) * PARTITION_SIZE;
    if (sinkEnd > csr->vertexNum) {
        sinkEnd = csr->vertexNum;
    }

    int blkStart = 0;
    rpa.push_back(0);
    for (int i = srcStart; i < srcEnd; i++) {
        int start = csr->rpao[i];
        int num = csr->rpao[i + 1] - csr->rpao[i];
        int blkNum = 0;
        for (int j = 0; j < num; j++) {
            int ngbVidx = csr->ciao[start + j];
            prop_t eProp = csr->ciao[start + j];
            if (ngbVidx >= sinkStart && ngbVidx < sinkEnd) {
                cia.push_back(ngbVidx);
                eProps.push_back(eProp);
                blkNum++;
            }
        }

        //update block csr data
        blkStart += blkNum;
        rpa.push_back(blkStart);
    }
    edgeNum = blkStart;
}
