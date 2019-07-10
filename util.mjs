const u = {}
// handles negative numbers safely
u.mod = (x, y) => {
	const m = x % y
	return m ? m : (m + y)
}
u.repeat = (x,f,d="") => (Array.isArray(x) ? x : Array.from(Array(x).keys())).map(f).join(d)
u.moore = (() => {
	const n = []
	for (var x = -1; x <= 1; ++x){
		for (var y = -1; y <= 1; ++y){
			if (x == 0 && y == 0) continue
			n.push([x,y])
		}
	}
	return n
})()
export default u