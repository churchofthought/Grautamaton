#version 310 es
${HEADER_INCLUDE("COMPUTE")}

layout( local_size_x = 1 ) in;

#define TRANSITION(dir) SET_CELL(dir ^ 1, C, transition( \
  GET_CELL(dir ^ 0, C), \
  uint[]( \
    ${u.repeat(c.NUM_NEIGHBORS, i => `GET_CELL(dir ^ 0, idxes[${i}])`, ',')} \
  ) \
));

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
    ${u.repeat(u.moore, ([x,y]) => `idx(x+uint(${x}),y+uint(${y}))`, ',')}
  );
  if ((time & uint(1)) == uint(1)){
    TRANSITION(1);
  }else{
    TRANSITION(0);
  }
}