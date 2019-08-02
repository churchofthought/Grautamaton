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

bool is_incoming(CELL_TYPE neighbor, ivec2 offs){
  if (neighbor.z == 0)
    return false;
    
  uint dist = max(max(uint(abs(neighbor.x)), uint(abs(neighbor.y))), uint(abs(neighbor.x + neighbor.y)));
  uint split = min(min(uint(abs(neighbor.x)), uint(abs(neighbor.y))), uint(abs(neighbor.x + neighbor.y)));

  uint moves_idx = (sign(neighbor.x) >= 0 ? 0u : 2u) + (sign(neighbor.y) >= 0 ? 0u : 1u);
  ivec2 move = time % dist >= split ? movements[moves_idx][0] : movements[moves_idx][1];
  return offs == move;
}



void add_cells(out ivec3 a, ivec3 b){
  //   // anti-matter
  //   if (a.z >= 1){
  //     a.xy *= -1;
  //   }

  //  // anti-matter
  //   if (b.z >= 1){
  //     b.xy *= -1;
  //   }

  int dist_a = max(1, max(max(abs(a.x), abs(a.y)), abs(a.x + a.y)));
  int dist_b = max(1, max(max(abs(b.x), abs(b.y)), abs(b.x + b.y)));

  // normalize each to their stepcount, and weight each by their mass
  a.xy *= (dist_b * a.z);
  b.xy *= (dist_a * b.z);

  // calculate sum
  a += b;

  // reduce the velocity to simplest form
  if (a.x != 0 && a.y != 0){
    a.xy /= get_gcd(a.x, a.y);
  }
}


CELL_TYPE transition(CELL_TYPE center, CELL_TYPE[NUM_NEIGHBORS] neighborhood){
  ${ts = ts || `
    CELL_TYPE v = ivec3(0, 0, 0);
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