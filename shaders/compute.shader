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

vec2 movements = {

}

bool is_incoming(vec2 velocity, vec2 offs){
  uint dist = max(abs(velocity.x), abs(velocity.y), abs(velocity.x + velocity.y));
  uint split = min(abs(velocity.x), abs(velocity.y), abs(velocity.x + velocity.y));
  
  uint move = time % dist >= split ? movement_1 : movement_2;
  // sign of x, sign of y
}

CELL_TYPE transition(CELL_TYPE center, CELL_TYPE[NUM_NEIGHBORS] neighborhood){
  ${ts = ts || `
    
    return center;
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