const u = {}
// handles negative numbers safely
u.mod = (x, y) => {
	const m = x % y
	return m ? m : (m + y)
}
u.repeat = (x,f,d="") => (Array.isArray(x) ? x : Array.from(Array(x).keys())).map(f).join(d)
u.neighborhoods = {}
u.neighborhoods.moore = () => {
	const n = []
	for (var x = -1; x <= 1; ++x){
		for (var y = -1; y <= 1; ++y){
			if (x == 0 && y == 0) continue
			n.push([x,y])
		}
	}
	return n
}
u.neighborhoods.hex = () => {
	const n = []
	for (var x = -1; x <= 1; ++x){
		for (var y = -1; y <= 1; ++y){
			if (x == y) continue
			n.push([x,y])
		}
	}
	return n
}

u.projectors = []
u.projectors.moore = (x,y) => [x,y]
u.projectors.hex = (x,y) => [2.0 / 3.0 * x, -1.0 / 3.0 * x + Math.sqrt(3.0) / 3.0 * y]


// u.maxXY = (neighborhood, w, h) => {
// 	const p = u.projectors[neighborhood]
// 	const vals = [
// 		p(0,0), p(w,0), p(0,h), p(w,h)
// 	]
// 	return [
// 		Math.ceil(Math.max(...vals.map(v => v[0]))), 
// 		Math.ceil(Math.max(...vals.map(v => v[1])))
// 	]
// }
export default u