#version 310 es
${HEADER_INCLUDE("COMPUTE")}

layout( local_size_x = 1 ) in;

#define TRANSITION(dir) ${u.cmacro(`
  SET_CELL(dir ^ 1, C, transition(  
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

bool is_incoming(CELL_TYPE neighbor, ivec2 offs){
  if (neighbor.z == 0)
    return false;
    
  ivec2 velocity = neighbor.xy * sign(neighbor.z);
  if (velocity == ivec2(0,0))
    return offs == ivec2(0,0);
    
  uint dist = max(max(uint(abs(velocity.x)), uint(abs(velocity.y))), uint(abs(velocity.x + velocity.y)));
  uint split = min(min(uint(abs(velocity.x)), uint(abs(velocity.y))), uint(abs(velocity.x + velocity.y)));

  uint moves_idx = (sign(velocity.x) >= 0 ? 0u : 2u) + (sign(velocity.y) >= 0 ? 0u : 1u);
  ivec2 move = time % dist >= split ? movements[moves_idx][0] : movements[moves_idx][1];
  return offs == move;
}



void add_cells(out CELL_TYPE a, CELL_TYPE b){
  int dist_a = max(1, max(max(abs(a.x), abs(a.y)), abs(a.x + a.y)));
  int dist_b = max(1, max(max(abs(b.x), abs(b.y)), abs(b.x + b.y)));

  // normalize each to their stepcount, and weight each by their mass (possibly negative mass)
  a.xy *= dist_b * a.z;
  b.xy *= dist_a * b.z;

  // calculate sum
  a += b;

  // reduce the velocity to simplest form
  if (a.x != 0 && a.y != 0){
    a.xy /= gcd(a.x, a.y);
  }
}


CELL_TYPE transition(CELL_TYPE[NUM_NEIGHBORS] neighborhood){
  ${ts = ts || `
    CELL_TYPE v;
    ${  
      u.repeat(c.NEIGHBORHOOD, ([x,y],i) => `
        if (is_incoming(neighborhood[${i}], ivec2(${-x},${-y}))){
          add_cells(v, neighborhood[${i}]);
        }
    `)}
    return v;
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