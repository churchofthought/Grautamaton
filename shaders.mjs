import c from "./constants.mjs"
import u from "./util.mjs"

export default async gl => {
	

	const header = (await (await fetch("./shaders/header.shader")).text()).replace(/\\/g,"\\\\")
	const HEADER_INCLUDE = type => eval(`\`${header}\``)

	const evalAsTemplate = (s) => eval(`\`${s.replace(/\\/g,"\\\\")}\``)
	const getShaderFile = async file => 
		[file, evalAsTemplate(await (await fetch(`./shaders/${file}.shader`)).text())]
	const shaderSources = Object.fromEntries(await Promise.all([
		"vertex", "fragment", "compute"
	].map(getShaderFile)))
	
	const lined = (s) => s.split(/[\r\n]+/).map((x, i) => `${i}. ${x}`).join("\n")
	console.log(
		lined(shaderSources.fragment) + "\n=====================================\n" + lined(shaderSources.compute)
	)

	const createShader = (program, sourceCode, type) => {
		// Compiles either a shader of type gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
		const shader = gl.createShader(gl[type])
		gl.shaderSource(shader, sourceCode)
		gl.compileShader(shader)

		if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
			throw `Could not compile WebGL ${type} shader. ${gl.getShaderInfoLog(shader)}`

		gl.attachShader(program, shader)

		return shader
	}

	const createProgram = shaders => {
		const program = gl.createProgram()

		for (const [source, type] of shaders)
			createShader(program, source, type)

		gl.linkProgram(program)

		if (!gl.getProgramParameter(program, gl.LINK_STATUS))
			throw `Could not compile WebGL program. ${gl.getProgramInfoLog(program)}`

		return program
	}

	return {
		renderProgram: createProgram([
			[shaderSources.vertex, "VERTEX_SHADER"],
			[shaderSources.fragment, "FRAGMENT_SHADER"]
		]),
		computeProgram: createProgram([
			[shaderSources.compute, "COMPUTE_SHADER"]
		])
	}
}