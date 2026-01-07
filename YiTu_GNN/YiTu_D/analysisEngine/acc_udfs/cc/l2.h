#ifndef __L2_H__
#define __L2_H__

#define prop_t unsigned int

#define MAX_PROP                  (INT_MAX - 1)

#define VERTEX_ACTIVE_BIT_MASK    (0x80000000)
#define IS_ACTIVE_VERTEX(a)       (a != 0)

/* source vertex property process */
inline prop_t preprocessProperty(prop_t srcProp)
{
    return ((srcProp));
}

/* source vertex property & edge property */
inline prop_t scatterFunc(prop_t srcProp, prop_t edgeProp)
{
    return (srcProp);
}

/* destination property update dst buffer update */
inline prop_t gatherFunc(prop_t ori, prop_t update)
{
    return (ori | update);
}

inline prop_t applyFunc( prop_t tProp,
                         prop_t source,
                         prop_t outDeg,
                         unsigned int arg
                       )
{
    return tProp | source;
}

#endif /* __L2_H__ */
