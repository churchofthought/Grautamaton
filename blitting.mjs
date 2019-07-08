export default (gl, program) => {
	


	const buffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer)


	const square = new Int8Array([
		-1, -1,
		-1, 1,
		1, -1,
		1, 1
	])

	gl.bufferData(gl.ARRAY_BUFFER, square, gl.STATIC_DRAW)

	const loc = gl.getAttribLocation(program, "position")

	gl.vertexAttribPointer(loc, 2, gl.BYTE, false, 0, 0)
	gl.enableVertexAttribArray(loc)
	

	//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
}