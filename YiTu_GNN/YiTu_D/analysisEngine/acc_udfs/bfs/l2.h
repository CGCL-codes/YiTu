#ifndef __L2_H__
#define __L2_H__

#define prop_t unsigned int
#include <climits>

#define MAX_PROP                  (INT_MAX - 1)

#define VERTEX_ACTIVE_BIT_MASK    (0x80000000)
#define IS_ACTIVE_VERTEX(a)       ((((((a) & VERTEX_ACTIVE_BIT_MASK) == VERTEX_ACTIVE_BIT_MASK))) ? 1 : 0)

/* source vertex property process */
inline prop_t preprocessProperty(prop_t srcProp)
{
  return ((srcProp) + 1);
}

/* source vertex property & edge property */
inline prop_t scatterFunc(prop_t srcProp, prop_t edgeProp)
{
  return (srcProp);
}

/* destination property update dst buffer update */
inline prop_t gatherFunc(prop_t ori, prop_t update)
{
  return (
           (
             (
               (((ori) & (~VERTEX_ACTIVE_BIT_MASK)) > ((update) & (~VERTEX_ACTIVE_BIT_MASK)))
               && (update != 0)
             )
             || (ori == 0x0)
           ) ? (update) : (ori)
         );
}

inline prop_t applyFunc( prop_t tProp,
                         prop_t source,
                         prop_t outDeg,
                         unsigned int arg
                       )
{
  prop_t update = 0;

  prop_t uProp  = source;
  prop_t wProp;
  if (((tProp & VERTEX_ACTIVE_BIT_MASK) == VERTEX_ACTIVE_BIT_MASK) && (uProp == MAX_PROP))
  {
    wProp = tProp; // current active vertex, not travsered
  }
  else
  {
    wProp = uProp & 0x7fffffff; // not travsered
  }
  update = wProp;

  return update;
}

#endif /* __L2_H__ */
