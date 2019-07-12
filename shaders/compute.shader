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
  uint blue = COUNT(neighborhood, BLUE);
  uint red = COUNT(neighborhood, RED);
  uint green = COUNT(neighborhood, GREEN);
  uint black = COUNT(neighborhood, BLACK);
  
  if (center == GREEN && red >= 1u)
    return BLACK;

  if (center == BLUE && green >= 1u)
    return BLACK;

  if (center == GREEN)
    return BLUE;

   

  if (center == BLUE)
    return RED;
  
  if (center == RED)
    return BLACK;

  if (red >= 1u || green >= 1u)
    return BLACK;

  if (blue >= 5u)
    return GREEN;
  
  if (blue >= 1u)
    return BLUE;


   
  // if (center == GREEN)
  //   return RED;
  // if (COUNT(neighborhood, GREEN) >= 1u && center == BLACK)
  //   return GREEN;	
  return BLACK;
}

void main(void){
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;

  uvec2 C = idx(x, y);
  uvec2[NUM_NEIGHBORS] idxes = uvec2[](
    ${u.repeat(u.neighborhoods[c.NEIGHBORHOOD](), ([x,y]) => `idx(x+uint(${x}),y+uint(${y}))`, ',')}
  );
  if ((time & 1u) == 1u){
    TRANSITION(1);
  }else{
    TRANSITION(0);
  }
}