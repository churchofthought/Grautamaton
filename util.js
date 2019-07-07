const u = {}
// handles negative numbers safely
u.mod = (x, y) => (x + y) % y
export default u