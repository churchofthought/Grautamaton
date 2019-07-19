const u = {}
// handles negative numbers safely
u.mod = (x, y) => {
	const m = x % y
	return Math.floor(m >= 0 ? m : (m + y))
}
u.repeat = (x,f,d="") => (Array.isArray(x) ? x : Array.from(Array(x).keys())).map(f).join(d)
u.cmacro = x => x.replace(/[\r\n]/g, "\\$&")


const searchParams = new URLSearchParams(location.search)
u.getSearchParam = x => searchParams.get(x)

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