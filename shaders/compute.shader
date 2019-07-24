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

ivec2[4][2] movements = ivec2[][](
  ivec2[](
    ivec2(+1,0), ivec2(0,+1)
  ),
  ivec2[](
    ivec2(+1,-1), ivec2(+1,0) 
  ),
  ivec2[](
    ivec2(-1,+1), ivec2(-1,0)
  ),
  ivec2[](
    ivec2(-1,0), ivec2(0,-1)
  )
);

bool is_incoming(ivec2 velocity, ivec2 offs){
  if (velocity == ivec2(0,0))
    return false;
    
  uint dist = max(max(uint(abs(velocity.x)), uint(abs(velocity.y))), uint(abs(velocity.x + velocity.y)));
  uint split = min(min(uint(abs(velocity.x)), uint(abs(velocity.y))), uint(abs(velocity.x + velocity.y)));

  uint moves_idx = (sign(velocity.x) >= 0 ? 0u : 2u) + (sign(velocity.y) >= 0 ? 0u : 1u);
  ivec2 move = time % dist >= split ? movements[moves_idx][0] : movements[moves_idx][1];
  return offs == move;
}

// CELL_TYPE add_cells(ivec4 a, ivec4 b){
//   // anti-matter
//   if (a.z >= 1){
//     a.xy *= -1;
//   }

//  // anti-matter
//   if (b.z >= 1){
//     b.xy *= -1;
//   }

//   // a.xy *= b.y
//   // b.xy *= a.y
// }


CELL_TYPE transition(CELL_TYPE center, CELL_TYPE[NUM_NEIGHBORS] neighborhood){
  ${ts = ts || `
    ${
      u.repeat(c.NEIGHBORHOOD, ([x,y],i) => `
        if (is_incoming(neighborhood[${i}], ivec2(${-x},${-y}))){
          return neighborhood[${i}];
        } 
    `)}
    return ivec2(0,0);
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