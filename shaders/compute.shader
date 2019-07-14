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
  uint red = COUNT(neighborhood, RED);
  uint blue = COUNT(neighborhood, BLUE);
  uint green = COUNT(neighborhood, GREEN);
  uint black = COUNT(neighborhood, BLACK);

  if (green >= 3u && red >= 1u && blue >= 1u)
    return GREEN;

  if (center == GREEN)
    return BLUE;
   
  if (center == BLUE)
    return RED;
  
  if (center == RED)
    return BLACK;

  if (green >= 1u)
    return GREEN;

  return BLACK;
}


void main(void){
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;

  uvec2 C = idx(x, y);
  uvec2[NUM_NEIGHBORS] idxes = uvec2[](
    ${u.repeat(c.NEIGHBORHOOD, ([x,y]) => `idx(omod(x,${x},uint(UNIVERSE_WIDTH)),omod(y,${y},uint(UNIVERSE_HEIGHT)))`, ',')}
  );
  if ((time & 1u) == 1u){
    TRANSITION(1);
  }else{
    TRANSITION(0);
  }
}