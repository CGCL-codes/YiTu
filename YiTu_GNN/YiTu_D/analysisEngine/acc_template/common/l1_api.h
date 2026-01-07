//提供了HLS流处理的基本API
#ifndef __L1_API_H__
#define __L1_API_H__

#include <ap_int.h>

#define CLEAR_CYCLE (256)

template <typename T>
inline int clear_stream (hls::stream<T> &stream)
{
#pragma HLS INLINE
    int end_counter = 0;
clear_stream: while (true)
    {
        T clear_data;

        if ( read_from_stream_nb(stream, clear_data) == 0)
        {
            end_counter ++;
        }
        if (end_counter > CLEAR_CYCLE)
        {
            break;
        }
    }
    return 0;
}

template <typename T>
inline int empty_stream (hls::stream<T> &stream)
{
#pragma HLS INLINE
    int end_counter = 0;
empty_stream: while (true)
    {
        T clear_data;

        if ( read_from_stream_nb(stream, clear_data) == 0)
        {
            end_counter ++;
        }
        else
        {
            end_counter = 0;
        }
        if (end_counter > 4096)
        {
            break;
        }
    }
    return 0;
}



template <typename T>
inline int write_to_stream (hls::stream<T> &stream, T const& value)
{
#pragma HLS INLINE
    int count = 0;
    stream << value;
    return 0;
}


template <typename T>
inline int read_from_stream (hls::stream<T> &stream, T & value)
{
#pragma HLS INLINE
    value = stream.read();
    return 0;
#if 0
    if (stream.read_nb(value))
    {
        return 0;
    }
    else
    {
        return -1;
    }
#endif
}


template <typename T>
inline int read_from_stream_nb (hls::stream<T> &stream, T & value)
{
#pragma HLS INLINE
    if (stream.empty())
    {
        return 0;
    }
    else
    {
        value = stream.read();
        return 1;
    }
}

#endif /* __L1_API_H__ */
