#version 310 es
${HEADER_INCLUDE("COMPUTE")}

layout( local_size_x = 1 ) in;

#define TRANSITION(dir) ${u.cmacro(`
  SET_CELL(dir ^ 1, C, transition( 
    GET_CELL(dir, C), 
    float[]( 
      ${u.repeat(c.NUM_NEIGHBORS, i => `GET_CELL(dir ^ 0, idxes[${i}])`, ',')} 
    ) 
  ));
`)}

float transition(float center, float[NUM_NEIGHBORS] neighborhood){
  ${ts = ts || `
    float e = pow(10.0, -30.0);
    float sum = ${u.repeat(c.NUM_NEIGHBORS, i => `(neighborhood[${i}] >= e ? (neighborhood[${i}] / 8.0) : 0.0)`, '+')};
    if (center >= e)
      center *= 1.0/8.0;
    return center + sum;
  `}
}


void main(void){
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;

  uint C = idx(x, y);
  uint[NUM_NEIGHBORS] idxes = uint[](
    ${u.repeat(c.NEIGHBORHOOD, ([x,y]) => `idx(omod(x,${x},uint(UNIVERSE_WIDTH)),omod(y,${y},uint(UNIVERSE_HEIGHT)))`, ',')}
  );
  if ((time & 1u) == 1u){
    TRANSITION(1);
  }else{
    TRANSITION(0);
  }
}