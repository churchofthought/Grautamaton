#version 310 es
${HEADER_INCLUDE("COMPUTE")}

layout( local_size_x = 1 ) in;

#define TRANSITION(dir) \
SET_CELL(dir ^ 1, C, \
  transition( \
    GET_CELL(dir ^ 0, C), \
    uint[]( \
      GET_CELL(dir ^ 0, idxes[0]), \
      GET_CELL(dir ^ 0, idxes[1]), \
      GET_CELL(dir ^ 0, idxes[2]), \
      GET_CELL(dir ^ 0, idxes[3]), \
      GET_CELL(dir ^ 0, idxes[4]), \
      GET_CELL(dir ^ 0, idxes[5]), \
      GET_CELL(dir ^ 0, idxes[6]), \
      GET_CELL(dir ^ 0, idxes[7]) \
    ) \
  ) \
);

uint transition(uint center, uint[NUM_NEIGHBORS] neighborhood){
  if (center == BLUE)
    return RED;
  if (COUNT(neighborhood, BLUE) >= uint(1) && center == BLACK)
    return BLUE;
  if (center == BLUE)
    return BLACK;
  if (center == RED)
    return BLACK;

  return BLACK;
}

void main(void){
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;

  uvec2 C = idx(x, y);

  uvec2[NUM_NEIGHBORS] idxes = uvec2[](
    idx(x-uint(1),y-uint(1)),
    idx(x-uint(1),y),
    idx(x-uint(1),y+uint(1)),
    idx(x,y-uint(1)),
    idx(x,y+uint(1)),
    idx(x+uint(1),y-uint(1)),
    idx(x+uint(1),y),
    idx(x+uint(1),y+uint(1))
  );
  if ((time & uint(1)) == uint(1)){
    TRANSITION(1);
  }else{
    TRANSITION(0);
  }
}