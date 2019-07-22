export default (gl, program) => {
	
	const buffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
	
	const loc = gl.getAttribLocation(program, "position")
	gl.enableVertexAttribArray(loc)
	gl.vertexAttribPointer(loc, 2, gl.BYTE, false, 0, 0)
	gl.bufferData(gl.ARRAY_BUFFER, new Int8Array([
		-1, -1,
		-1, 1,
		1, -1,
		1, 1
	]), gl.STATIC_DRAW)
}