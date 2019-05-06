export default (gl, program) => {
	


	const buffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer)

	
	const hexagon = [0,0].concat([...Array(7)].map((x,i) => [
		Math.cos(i*Math.PI/3),
		Math.sin(i*Math.PI/3)
	]).flat())

	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(hexagon), gl.STATIC_DRAW)

	const loc = gl.getAttribLocation(program, "position")

	gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
	gl.enableVertexAttribArray(loc)
	

	//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
}