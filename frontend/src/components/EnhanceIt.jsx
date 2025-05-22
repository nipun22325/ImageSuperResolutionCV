import { useState, useRef } from 'react';
import React from 'react';
import './EnhanceIt.css';

export default function EnhanceIt() {
  const [image, setImage] = useState(null);
  const [enhancedImage, setEnhancedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef();

  const handleDrop = (e) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleFileBrowse = (e) => {
    handleFiles(e.target.files);
  };

  const handleFiles = (files) => {
    if (files && files[0]) {
      const selectedFile = files[0];
      const reader = new FileReader();
      
      reader.onload = (e) => {
        setImage({
          file: selectedFile,
          preview: e.target.result
        });
        setEnhancedImage(null); // Reset enhanced image when new image is uploaded
      };
      
      reader.readAsDataURL(selectedFile);
    }
  };

  const enhanceImage = async () => {
    if (!image) return;
  
    setIsProcessing(true);
    
    try {
      const formData = new FormData();
      formData.append('file', image.file);
      
      const response = await fetch('http://localhost:8000/enhance/', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Enhancement failed');
      }
  
      // Get the enhanced image blob
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
  
      // Trigger download automatically
      const link = document.createElement('a');
      link.href = url;
      link.download = `enhanced_${image.file.name}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
  
      // Set the enhanced image for preview
      setEnhancedImage(url);
      
      // Immediately reset the image and file input after download
      setImage(null);
      setEnhancedImage(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = ''; // Reset the file input
      }
      
    } catch (error) {
      console.error('Error enhancing image:', error);
      alert('Failed to enhance image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="enhance-container">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <img src="/magic-wand.png" alt="Logo" className="logo-image" />
          <span className="logo-text">EnhanceIt</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="content-box">
          <h1 className="title">Enhance Your Images</h1>
          <p className="subtitle">Upload your low-resolution satellite images and enhance them with just a click.</p>
          
          {/* Upload Area */}
          <div 
            className={`upload-area ${image ? 'has-image' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            {image ? (
              <div className="upload-content">
                <img 
                  src={enhancedImage || image.preview} 
                  alt="Uploaded" 
                  className="uploaded-image"
                />
              </div>
            ) : (
              <div className="upload-instructions">
                <p className="upload-text">Drag & Drop or Browse</p>
                <input 
                  type="file" 
                  ref={fileInputRef}
                  onChange={handleFileBrowse}
                  accept="image/*"
                  className="hidden"
                  style={{ display: 'none' }}
                />
                <button 
                  onClick={() => fileInputRef.current.click()}
                  className="browse-button"
                >
                  Select file
                </button>
              </div>
            )}
          </div>
          
          {/* Action Buttons */}
          {image && (
            <div className="action-buttons">
              <button 
                onClick={enhanceImage}
                disabled={!image || isProcessing}
                className="action-button"
              >
                {isProcessing ? 'Enhancing...' : 'Enhance'}
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}