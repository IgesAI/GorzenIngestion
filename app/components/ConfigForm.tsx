'use client'

import React, { useState } from 'react'
import { ArrowLeft, Database, Zap, Settings, Eye, EyeOff } from 'lucide-react'

interface ConfigFormProps {
  fileCount: number
  onSubmit: (config: any) => void
  onBack: () => void
}

export default function ConfigForm({ fileCount, onSubmit, onBack }: ConfigFormProps) {
  const [config, setConfig] = useState({
    indexName: '',
    pineconeApiKey: '',
    openaiApiKey: '',
    useOpenAI: false,
    chunkSize: 'medium',
    enrichments: {
      code: false,
      formula: false,
      pictureClasses: false,
      pictureDescription: false
    },
    cloud: 'aws',
    region: 'us-east-1'
  })

  const [showPineconeKey, setShowPineconeKey] = useState(false)
  const [showOpenAIKey, setShowOpenAIKey] = useState(false)
  const [errors, setErrors] = useState<{ [key: string]: string }>({})

  const validateForm = () => {
    const newErrors: { [key: string]: string } = {}

    if (!config.indexName.trim()) {
      newErrors.indexName = 'Index name is required'
    } else if (!/^[a-z0-9-]+$/.test(config.indexName)) {
      newErrors.indexName = 'Index name must contain only lowercase letters, numbers, and hyphens'
    }

    if (!config.pineconeApiKey.trim()) {
      newErrors.pineconeApiKey = 'Pinecone API key is required'
    }

    if (config.useOpenAI && !config.openaiApiKey.trim()) {
      newErrors.openaiApiKey = 'OpenAI API key is required when using OpenAI embeddings'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (validateForm()) {
      onSubmit(config)
    }
  }

  const generateIndexName = () => {
    const timestamp = Date.now().toString(36)
    const random = Math.random().toString(36).substring(2, 8)
    setConfig(prev => ({ ...prev, indexName: `docs-${timestamp}-${random}` }))
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center mb-6">
        <button
          onClick={onBack}
          className="flex items-center text-gray-600 hover:text-gray-900 transition-colors mr-4"
        >
          <ArrowLeft className="h-5 w-5 mr-1" />
          Back
        </button>
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Configure Processing</h2>
          <p className="text-gray-600">Set up your Pinecone index and processing options</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Pinecone Configuration */}
        <div className="bg-gray-50 rounded-lg p-6">
          <div className="flex items-center mb-4">
            <Database className="h-6 w-6 text-blue-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Pinecone Configuration</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Index Name *
              </label>
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={config.indexName}
                  onChange={(e) => setConfig(prev => ({ ...prev, indexName: e.target.value }))}
                  placeholder="my-vector-index"
                  className={`flex-1 border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                    errors.indexName ? 'border-red-500' : 'border-gray-300'
                  }`}
                />
                <button
                  type="button"
                  onClick={generateIndexName}
                  className="bg-gray-200 text-gray-700 px-3 py-2 rounded-lg hover:bg-gray-300 transition-colors text-sm"
                >
                  Generate
                </button>
              </div>
              {errors.indexName && (
                <p className="text-red-600 text-sm mt-1">{errors.indexName}</p>
              )}
              <p className="text-xs text-gray-500 mt-1">
                Lowercase letters, numbers, and hyphens only
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Pinecone API Key *
              </label>
              <div className="relative">
                <input
                  type={showPineconeKey ? 'text' : 'password'}
                  value={config.pineconeApiKey}
                  onChange={(e) => setConfig(prev => ({ ...prev, pineconeApiKey: e.target.value }))}
                  placeholder="Enter your Pinecone API key"
                  className={`w-full border rounded-lg px-3 py-2 pr-10 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                    errors.pineconeApiKey ? 'border-red-500' : 'border-gray-300'
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowPineconeKey(!showPineconeKey)}
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                >
                  {showPineconeKey ? (
                    <EyeOff className="h-4 w-4 text-gray-400" />
                  ) : (
                    <Eye className="h-4 w-4 text-gray-400" />
                  )}
                </button>
              </div>
              {errors.pineconeApiKey && (
                <p className="text-red-600 text-sm mt-1">{errors.pineconeApiKey}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Cloud Provider
              </label>
              <select
                value={config.cloud}
                onChange={(e) => setConfig(prev => ({ ...prev, cloud: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                aria-label="Cloud Provider"
              >
                <option value="aws">AWS</option>
                <option value="gcp">Google Cloud</option>
                <option value="azure">Azure</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Region
              </label>
              <select
                value={config.region}
                onChange={(e) => setConfig(prev => ({ ...prev, region: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                aria-label="Region"
              >
                {config.cloud === 'aws' && (
                  <>
                    <option value="us-east-1">US East (N. Virginia)</option>
                    <option value="us-west-2">US West (Oregon)</option>
                    <option value="eu-west-1">Europe (Ireland)</option>
                  </>
                )}
                {config.cloud === 'gcp' && (
                  <>
                    <option value="us-central1">US Central</option>
                    <option value="us-east1">US East</option>
                    <option value="europe-west1">Europe West</option>
                  </>
                )}
                {config.cloud === 'azure' && (
                  <>
                    <option value="eastus">East US</option>
                    <option value="westus2">West US 2</option>
                    <option value="westeurope">West Europe</option>
                  </>
                )}
              </select>
            </div>
          </div>
        </div>

        {/* Embedding Configuration */}
        <div className="bg-gray-50 rounded-lg p-6">
          <div className="flex items-center mb-4">
            <Zap className="h-6 w-6 text-green-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Embedding Configuration</h3>
          </div>

          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="useOpenAI"
                checked={config.useOpenAI}
                onChange={(e) => setConfig(prev => ({ ...prev, useOpenAI: e.target.checked }))}
                className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              />
              <label htmlFor="useOpenAI" className="text-sm font-medium text-gray-700">
                Use OpenAI Embeddings (text-embedding-3-small)
              </label>
            </div>
            
            <p className="text-xs text-gray-500 ml-6">
              {config.useOpenAI 
                ? 'Higher quality embeddings, requires OpenAI API key' 
                : 'Free local embeddings using BAAI/bge-small-en-v1.5'
              }
            </p>

            {config.useOpenAI && (
              <div className="ml-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  OpenAI API Key *
                </label>
                <div className="relative">
                  <input
                    type={showOpenAIKey ? 'text' : 'password'}
                    value={config.openaiApiKey}
                    onChange={(e) => setConfig(prev => ({ ...prev, openaiApiKey: e.target.value }))}
                    placeholder="Enter your OpenAI API key"
                    className={`w-full border rounded-lg px-3 py-2 pr-10 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                      errors.openaiApiKey ? 'border-red-500' : 'border-gray-300'
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowOpenAIKey(!showOpenAIKey)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  >
                    {showOpenAIKey ? (
                      <EyeOff className="h-4 w-4 text-gray-400" />
                    ) : (
                      <Eye className="h-4 w-4 text-gray-400" />
                    )}
                  </button>
                </div>
                {errors.openaiApiKey && (
                  <p className="text-red-600 text-sm mt-1">{errors.openaiApiKey}</p>
                )}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chunk Size
              </label>
              <select
                value={config.chunkSize}
                onChange={(e) => setConfig(prev => ({ ...prev, chunkSize: e.target.value }))}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                aria-label="Chunk Size"
              >
                <option value="small">Small (512 tokens) - Precise search</option>
                <option value="medium">Medium (1024 tokens) - Balanced</option>
                <option value="large">Large (1536 tokens) - More context</option>
              </select>
            </div>
          </div>
        </div>

        {/* Advanced Options */}
        <div className="bg-gray-50 rounded-lg p-6">
          <div className="flex items-center mb-4">
            <Settings className="h-6 w-6 text-purple-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Advanced Processing Options</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="enrichCode"
                  checked={config.enrichments.code}
                  onChange={(e) => setConfig(prev => ({
                    ...prev,
                    enrichments: { ...prev.enrichments, code: e.target.checked }
                  }))}
                  className="rounded border-gray-300 text-blue-600"
                />
                <label htmlFor="enrichCode" className="text-sm text-gray-700">
                  Code understanding
                </label>
              </div>
              
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="enrichFormula"
                  checked={config.enrichments.formula}
                  onChange={(e) => setConfig(prev => ({
                    ...prev,
                    enrichments: { ...prev.enrichments, formula: e.target.checked }
                  }))}
                  className="rounded border-gray-300 text-blue-600"
                />
                <label htmlFor="enrichFormula" className="text-sm text-gray-700">
                  Mathematical formulas
                </label>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="enrichPictureClasses"
                  checked={config.enrichments.pictureClasses}
                  onChange={(e) => setConfig(prev => ({
                    ...prev,
                    enrichments: { ...prev.enrichments, pictureClasses: e.target.checked }
                  }))}
                  className="rounded border-gray-300 text-blue-600"
                />
                <label htmlFor="enrichPictureClasses" className="text-sm text-gray-700">
                  Image classification
                </label>
              </div>
              
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="enrichPictureDescription"
                  checked={config.enrichments.pictureDescription}
                  onChange={(e) => setConfig(prev => ({
                    ...prev,
                    enrichments: { ...prev.enrichments, pictureDescription: e.target.checked }
                  }))}
                  className="rounded border-gray-300 text-blue-600"
                />
                <label htmlFor="enrichPictureDescription" className="text-sm text-gray-700">
                  Image descriptions
                </label>
              </div>
            </div>
          </div>
          
          <p className="text-xs text-gray-500 mt-3">
            Enrichments improve search quality but increase processing time
          </p>
        </div>

        {/* Summary */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 mb-2">Processing Summary</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• {fileCount} files will be processed</li>
            <li>• Index "{config.indexName || 'your-index'}" will be created in Pinecone</li>
            <li>• {config.useOpenAI ? 'OpenAI' : 'Local'} embeddings will be used</li>
            <li>• Chunk size: {config.chunkSize}</li>
            {Object.values(config.enrichments).some(Boolean) && (
              <li>• Advanced enrichments enabled</li>
            )}
          </ul>
        </div>

        <div className="flex justify-end">
          <button
            type="submit"
            className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Start Processing
          </button>
        </div>
      </form>
    </div>
  )
}
