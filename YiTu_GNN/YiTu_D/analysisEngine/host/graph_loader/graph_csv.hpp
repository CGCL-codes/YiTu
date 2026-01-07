#ifndef __HOST_GRAPH_CSV_HPP__
#define __HOST_GRAPH_CSV_HPP__

#include "he_mem.h"
#include "he_mem_id.h"

#include "host_graph_data_structure.h"

extern std::vector<he_mem_t*> allocate_he_mem;

//从CSV文件加载数据到FPGA设备内存，内存管理通过he_mem_t结构实现，自动处理内存分配和数据传输
template <typename T>
T * load_from_csv(std::string file_name, int he_id, int mem_id)
{
    T data;
    //DEBUG_PRINTF("id: %d\n", he_id);
    std::vector<T> load_buffer;
    std::ifstream fhandle(file_name.c_str());
    if (!fhandle.is_open()) {
        DEBUG_PRINTF("error: can not open %s \n", file_name.c_str());
        return NULL;
    }
    int tmp_cnt = 0;
    while (fhandle.peek()  != EOF )
    {
        tmp_cnt ++;
        fhandle >> data;
        load_buffer.push_back(data);
    }
    fhandle.close();

    he_mem_t *p_mem = get_he_mem(he_id);
    // create new, if not existing
    if (p_mem == NULL)
    {
        he_mem_t *mem = (he_mem_t *)memalign( 4, sizeof(he_mem_t));
        mem->id =  he_id;
        mem->name = "load";
        mem->attr = mem_id;
        mem->unit_size = sizeof(T) * load_buffer.size();
        mem->size_attr = SIZE_USER_DEFINE;

        he_mem_init(getAccelerator()->context, mem);
        allocate_he_mem.push_back(mem);
    }

    p_mem = get_he_mem(he_id);
    if (p_mem->size <  sizeof(T) * load_buffer.size())
    {
        DEBUG_PRINTF("    warning mem %d is too small\n", he_id);
    }
    int load_size = (p_mem->size < sizeof(T) * load_buffer.size()) ?
                    (p_mem->size / sizeof(T)) : load_buffer.size();
    for (int i = 0; i < load_size; i++)
    {
        (((T*)p_mem->data)[i]) = load_buffer[i];
    }
    int id = he_id;
    //DEBUG_PRINTF("size %d \n", p_mem->size);
    transfer_data_to_pl(getAccelerator()->context, getAccelerator()->device, &id, 1);
    return (T*)p_mem->data;
}


//初始化输出内存区域
template <typename T>
int output_init(int he_id, int mem_id, int ref_he_id)
{
    he_mem_t *p_ref = get_he_mem(ref_he_id);
    if (p_ref == NULL)
    {
        return -1;
    }
    he_mem_t *p_mem = get_he_mem(he_id);
    // create new, if not existing
    if (p_mem == NULL)
    {
        he_mem_t *mem = (he_mem_t *)memalign( 4, sizeof(he_mem_t));
        mem->id =  he_id;
        mem->name = "output";
        mem->attr = mem_id;
        mem->unit_size = p_ref->size;
        mem->size_attr = SIZE_USER_DEFINE;
        he_mem_init(getAccelerator()->context, mem);
        allocate_he_mem.push_back(mem);
    }

    p_mem = get_he_mem(he_id);
    int load_size = (p_mem->size) / sizeof(T);
    for (int i = 0; i < load_size; i++)
    {
        (((T*)p_mem->data)[i]) = 0;
    }
    return 0;
}

//将FPGA计算结果写回CSV文件
template <typename T>
int write_back_csv(std::string file_name, int he_id)
{
    std::vector<T> load_buffer;
    std::ofstream fhandle(file_name.c_str());
    if (!fhandle.is_open()) {
        DEBUG_PRINTF("error: can not open %s \n", file_name.c_str());
        exit(EXIT_FAILURE);
    }

    transfer_data_from_pl(getAccelerator()->context, getAccelerator()->device, he_id);
    const he_mem_t *p_mem = get_he_mem(he_id);
    const  T * p_data = (T *)(p_mem->data);

    for (unsigned int i = 0; i < p_mem->size / sizeof(T); i++)
    {
        fhandle << p_data[i] << std::endl;
    }
    fhandle.flush();
    fhandle.close();

    return 0;
}

#endif /* __HOST_GRAPH_CSV_HPP__ */
