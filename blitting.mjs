export default (gl, program) => {
	const buffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer)

	gl.bufferData(gl.ARRAY_BUFFER, new Int8Array([
		-1, -1,
		-1, 1,
		1, -1,
		1, 1
	]), gl.STATIC_DRAW)

	const loc = gl.getAttribLocation(program, "vertexCoords")

	gl.vertexAttribPointer(loc, 2, gl.BYTE, false, 0, 0)
	gl.enableVertexAttribArray(loc)
}