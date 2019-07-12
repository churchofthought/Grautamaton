import c from "./constants.mjs"

export default (gl, program) => {
	


	const buffer = gl.createBuffer()
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
	
	
	const loc = gl.getAttribLocation(program, "position")
	gl.enableVertexAttribArray(loc)
	const shape = (() => {
		switch (c.NEIGHBORHOOD){
		case "moore":
			gl.vertexAttribPointer(loc, 2, gl.BYTE, false, 0, 0)
			return new Int8Array([
				-1, -1,
				-1, 1,
				1, -1,
				1, 1
			])
		case "hex":
			gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
			return new Float32Array([0,0].concat([...Array(7)].map((x,i) => [
				Math.cos(i*Math.PI/3),
				Math.sin(i*Math.PI/3)
			]).flat()))
		}
	})()

	gl.bufferData(gl.ARRAY_BUFFER, shape, gl.STATIC_DRAW)
	

	//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
}