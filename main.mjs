import createPrograms from "./shaders.mjs"
import setupBlitting from "./blitting.mjs"
import constants from "./constants.mjs"
import util from "./util.mjs"

// window.onresize = () => {

// }

window.onload = async () => {

	const container = document.createElement("div")

	const canvas = document.createElement("canvas")
	canvas.width = constants.CANVAS_WIDTH
	canvas.height = constants.CANVAS_HEIGHT


	const status = document.createElement("div")

	// set styles
	document.documentElement.style = document.body.style = `
		padding: 0;
		margin: 0;
		margin-top: 2em;
		text-align: center;
	`

	container.style = `
		display:inline-block;
		position: relative;
		margin-left: auto;
		margin-right: auto;
	`

	status.style = `
		position: absolute;
		left: 0;
		top: 0;
		margin: 0.5em;
		z-index: 1;
		color: red;
		font-family: monospace;
		font-weight: bold;
		font-size: 1em;
		pointer-events: none;
	`

	container.appendChild(canvas)
	container.appendChild(status)
	document.body.appendChild(container)

	const gl = canvas.getContext("webgl2-compute", {
		alpha: false, 
		depth: false, 
		stencil: false, 
		antialias: false, 
		preserveDrawingBuffer: false, 
		powerPreference: "high-performance", 
		failIfMajorPerformanceCaveat: true,
		desynchronized: true
	})
	// Only continue if WebGL is available and working
	if (!gl)
		throw "Unable to initialize WebGL2 Compute. Your browser or machine may not support it."

	// gl.viewport(0, 0, canvas.width, canvas.height)
	// gl.clearColor(0, 0.5, 0, 1)
	// gl.clear(gl.COLOR_BUFFER_BIT)

	const { renderProgram, computeProgram } = await createPrograms(gl)

	setupBlitting(gl, renderProgram)

	const mousePosToArr = (rawX, rawY) => {
		const [x, y] = constants.PROJECTOR(
			rawX, 
			constants.CANVAS_HEIGHT - rawY
		)

		const u = util.mod(x, constants.UNIVERSE_WIDTH)
		const v = util.mod(y, constants.UNIVERSE_HEIGHT)
		
		return u * constants.UNIVERSE_HEIGHT + v
	}

	const bindUniverse = idx => {
		const buffer = gl.createBuffer()
		gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, buffer)

		const arr = new Int32Array(constants.UNIVERSE_INT_SIZE)

		// for (var x = 0; x < constants.CANVAS_WIDTH; x++)
		// 	for (var y = 0; y < constants.CANVAS_HEIGHT; y++) {
		// 		arr[2 * mousePosToArr(x, y)] = x - constants.CANVAS_WIDTH / 2
		// 		arr[2 * mousePosToArr(x, y) + 1] = y - constants.CANVAS_HEIGHT / 2
		// 	}

		arr[2 * mousePosToArr(constants.CANVAS_WIDTH/2, constants.CANVAS_HEIGHT/2)] = 0
		arr[2 * mousePosToArr(constants.CANVAS_WIDTH/2, constants.CANVAS_HEIGHT/2) + 1] = 0
		arr[2 * mousePosToArr(constants.CANVAS_WIDTH/2, constants.CANVAS_HEIGHT/2) + 2] = -1
		
		// for (var i = constants.UNIVERSE_FLOAT_SIZE; i--;){
		// 	arr[i] = 2 * Math.random() - 1
		// }

		//arr[constants.UNIVERSE_SIZE - 1] = 20000
		// for (var x = 0; x < constants.UNIVERSE_WIDTH; x++)
		// 	for (var y = 0; y < constants.UNIVERSE_HEIGHT; y++)
		// 		arr[x * constants.UNIVERSE_HEIGHT + y] = -1.0

		//arr[0] = 1

		// for (var i = constants.UNIVERSE_INT_SIZE; i--;){
		// 	arr[i] = (Math.random() > 0.5 ? 1 : -1) * Math.round(Math.random() * 1) 
		// }

		gl.bufferData(gl.SHADER_STORAGE_BUFFER, arr /* constants.UNIVERSE_BYTE_SIZE */, gl.DYNAMIC_COPY)
		gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, idx, buffer)
	}

	const bindRenderMeta = () => {
		const buffer2 = gl.createBuffer()
		gl.bindBuffer(gl.UNIFORM_BUFFER, buffer2)
		gl.bufferData(gl.UNIFORM_BUFFER, 16, gl.DYNAMIC_DRAW)
		gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, buffer2)
		//gl.bindBufferRange(gl.UNIFORM_BUFFER, 2, buffer, 0, 16)
		// for (const x of [computeProgram, renderProgram])
		// 	gl.uniformBlockBinding(x, 0, 3)
	}


	
	[1, 0].forEach(bindUniverse)
	bindRenderMeta()

	var time = new Uint32Array([0])
	var startTime = Date.now() / 1000

	const render = () => {

		//render
		// if (time % 100 == 0){
			gl.useProgram(renderProgram)
			gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
		// }

		// ++time
		status.textContent = `${(++time[0] / (Date.now() / 1000 - startTime)).toFixed(2)} fps`
		gl.bufferSubData(gl.UNIFORM_BUFFER, 0, time)
		//gl.memoryBarrier(gl.SHADER_STORAGE_BARRIER_BIT);
		
		// compute next frame
		gl.useProgram(computeProgram)
		gl.dispatchCompute(constants.UNIVERSE_WIDTH, constants.UNIVERSE_HEIGHT, 1)
		
		requestAnimationFrame(render)
	}

	render()

	const universeView = new Int32Array(1)
	var offsetX, offsetY, shiftKey, button
	
	const updateMouse = event => {
		({shiftKey, offsetX, offsetY } = event)
	}

	const mousedownHandler = () => {
		const offset = constants.CELL_BITS * mousePosToArr(offsetX, offsetY)
		const byteOffset = offset / 8
		gl.getBufferSubData(gl.SHADER_STORAGE_BUFFER, byteOffset, universeView)

		const orig = universeView[0]

		//clear current value 
		universeView[0] = 0

		const newVal = (() => {
			switch (button) {

			// left click
			case 0:
				return 0

			// right click
			case 1:
				return 1
	
			// middle click
			case 2:
				return 2

			// ????
			default:
				return 3
			}
		})()

		universeView[0] = shiftKey ? orig - newVal : newVal

		gl.bufferSubData(gl.SHADER_STORAGE_BUFFER, byteOffset, universeView)
	}
	var mousedownInterval = 0
	canvas.onmousemove = updateMouse
	canvas.oncontextmenu = e => e.preventDefault()
	canvas.onmousedown = e => {
		({ button, offsetX, offsetY } = e)
		if (mousedownInterval) clearInterval(mousedownInterval)
		mousedownInterval = setInterval(mousedownHandler, 0)
		e.preventDefault()
	}
	window.onmouseup = () => {
		if (mousedownInterval) clearInterval(mousedownInterval)
		mousedownInterval = 0
	}
}


