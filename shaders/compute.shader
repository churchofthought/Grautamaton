#version 310 es
${HEADER_INCLUDE("COMPUTE")}

layout( local_size_x = 1 ) in;

#define TRANSITION(dir) ${u.cmacro(`
  SET_CELL(dir ^ 1, C, transition( 
    GET_CELL(dir, C), 
    CELL_TYPE[]( 
      ${u.repeat(c.NUM_NEIGHBORS, i => `GET_CELL(dir ^ 0, idxes[${i}])`, ',')} 
    ) 
  ));
`)}


vec2 rotate(vec2 v, float a) {
	float s = sin(a);
	float c = cos(a);
	mat2 m = mat2(c, -s, s, c);
	return m * v;
}

CELL_TYPE transition(CELL_TYPE center, CELL_TYPE[NUM_NEIGHBORS] neighborhood){
  ${ts = ts || `
    float e = 0.000000001;
    float ratio = 1.0/400.0;
    CELL_TYPE sum = ${u.repeat(c.NUM_NEIGHBORS, i => `(zAbs(neighborhood[${i}]) >= e ? ((1.0 - ratio) * rotate(neighborhood[${i}], 3.14/2.0) / float(NUM_NEIGHBORS)) : vec2(0.0,0.0))`, '+')};
    if (zAbs(center) >= e)
      center *= ratio;
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