export async function loadShader(path: string): Promise<string> {
  try {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const text = await response.text();
    if (!text) {
      throw new Error('Shader file is empty');
    }
    return text;
  } catch (error) {
    console.error(`Failed to load shader from ${path}:`, error);
    throw error;
  }
}

// Process imports in shader code
export async function processShaderImports(
  shaderCode: string,
  basePath: string = '/shaders/'
): Promise<string> {
  // Regular expression to match @import statements
  const importRegex = /@import\s+["']([^"']+)["']\s*;/g;

  let processedCode = shaderCode;
  let imports: RegExpExecArray | null;

  // Keep track of processed imports to avoid circular dependencies
  const processedImports = new Set<string>();

  while ((imports = importRegex.exec(shaderCode)) !== null) {
    const [fullMatch, importPath] = imports;

    // Avoid circular imports
    if (processedImports.has(importPath)) {
      processedCode = processedCode.replace(fullMatch, '// Import already processed');
      continue;
    }

    processedImports.add(importPath);

    // Determine the full path of the import
    const fullImportPath = importPath.startsWith('/') ? importPath : `${basePath}${importPath}`;

    try {
      // Load and process the imported shader
      const importedCode = await loadShader(fullImportPath);
      const processedImport = await processShaderImports(importedCode, basePath);

      // Replace the import statement with the imported code
      processedCode = processedCode.replace(fullMatch, processedImport);
    } catch (error) {
      console.error(`Failed to process import ${importPath}:`, error);
      // Replace with comment about failed import
      processedCode = processedCode.replace(
        fullMatch,
        `// Failed to import ${importPath}: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  return processedCode;
}

export async function loadShaderModules(
  device: GPUDevice,
  paths: Record<string, string>,
  shaderHeader: string,
  min16floatStorage: string
): Promise<Record<string, GPUShaderModule>> {
  const modules: Record<string, GPUShaderModule> = {};

  const shaderPromises = Object.entries(paths).map(async ([name, path]) => {
    try {
      const shaderCode = await loadShader(path);
      // Process imports before creating module
      var processedCode = await processShaderImports(shaderCode);

      //replace all instances of min16float_storage in shader code with min16floatStorage
      processedCode = processedCode.replace(/min16float_storage/g, min16floatStorage);

      modules[name] = device.createShaderModule({
        code: shaderHeader + `\n` + processedCode,
        label: `Shader module: ${name}`,
      });
    } catch (error) {
      console.error(`Failed to load shader module ${name} from ${path}:`, error);
      throw error;
    }
  });

  await Promise.all(shaderPromises);
  return modules;
}
