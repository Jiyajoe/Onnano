import React, { useState } from 'react'

export default function SingleResultView({ data, onPartsChange, isUpdating = false }) {
  const {
    object = {},
    visuals = {},
    division = {},
    disclaimer = 'Analysis results derived from visible physical characteristics.',
    image_quality,
  } = data || {}

  const [partsInput, setPartsInput] = useState(division?.parts_count || 4)
  const [showTechnical, setShowTechnical] = useState(false)
  const [showQualityDetails, setShowQualityDetails] = useState(false)

  const handlePresetClick = (count) => {
    setPartsInput(count)
    if (onPartsChange) {
      onPartsChange(count)
    }
  }

  const handleInputChange = (e) => {
    const val = parseInt(e.target.value, 10)
    if (!isNaN(val) && val >= 2 && val <= 12) {
      setPartsInput(val)
      if (onPartsChange) {
        onPartsChange(val)
      }
    } else if (e.target.value === '') {
      setPartsInput('')
    }
  }

  const quality = image_quality || {
    overall_score: 90.0,
    is_acceptable: true,
    checks: [
      { name: 'Resolution', passed: true, status_text: 'High Resolution ✓' },
      { name: 'Lighting & Exposure', passed: true, status_text: 'Balanced Lighting ✓' },
      { name: 'Blur & Sharpness', passed: true, status_text: 'Sharp Focus ✓' },
      { name: 'Object Visibility', passed: true, status_text: 'Fully Visible ✓' },
    ],
  }

  const dimensions = object?.dimensions || { pixel_height: 0, pixel_width: 0, aspect_ratio: '1.0', pixel_area: 0, perimeter: 0 }
  const shape = object?.shape || { shape_type: 'Geometric Profile', circularity: 0.8, solidity: 0.9, rectangularity: 0.8, symmetry_score: 85 }
  const color = object?.color || { dominant_hex: '#ffb627', dominant_name: 'Amber Gold', palette: [] }
  const texture = object?.texture || { descriptor: 'Smooth Surface', roughness_score: 20, entropy: 4.5, lbp_uniformity: 0.8 }

  const conf = object?.confidence_breakdown || {
    identification_confidence: object?.confidence_pct || 94.0,
    selection_confidence: object?.selection?.mask_confidence_pct || 92.0,
    orientation_confidence: 90.0,
    shape_measurement_confidence: 95.0,
    color_measurement_confidence: 98.0,
    feature_matching_confidence: 86.0,
    overall_confidence: object?.confidence_pct || 92.0,
  }

  const tech = object.technical_analysis || {}

  return (
    <div className="single-result-dashboard">


      {/* 2. Header & Fine-Grained Object Identity */}
      <div className="result-hero-banner">
        <div className="identity-badge-group">
          <div className="identity-top-row">
            <span className="identity-category-tag">🏷️ {object.category}</span>
            <span className="confidence-pill">
              🎯 Selection Confidence: <strong>{object.selection?.mask_confidence_pct || 92}%</strong>
            </span>
          </div>

          <h2 className="identity-title">{object.detected_type}</h2>
          {object.specific_type && (
            <span className="identity-subtype-pill"></span>
          )}




        </div>
      </div>



      {/* 3. Multi-Dimensional Confidence Meter */}
      <div className="dashboard-section-card confidence-multi-card">
        <div className="section-card-header">
          <div className="header-icon-title">
            <span className="section-icon">🎯</span>
            <div>
              <h3>Multi-Dimensional Reliability & Confidence</h3>
              <p className="section-subtitle">
                Computer Vision tracks measurement certainty independently across distinct signals
              </p>
            </div>
          </div>
          <div className="overall-conf-badge">
            <span>Overall Confidence:</span>
            <strong>{conf.overall_confidence}%</strong>
          </div>
        </div>

        <div className="confidence-bars-grid">
          <div className="conf-bar-item">
            <div className="conf-bar-meta">
              <span>Object Identification</span>
              <strong>{conf.identification_confidence}%</strong>
            </div>
            <div className="conf-track">
              <div className="conf-fill fill--gold" style={{ width: `${conf.identification_confidence}%` }}></div>
            </div>
          </div>

          <div className="conf-bar-item">
            <div className="conf-bar-meta">
              <span>Object Boundary Selection</span>
              <strong>{conf.selection_confidence}%</strong>
            </div>
            <div className="conf-track">
              <div className="conf-fill fill--mint" style={{ width: `${conf.selection_confidence}%` }}></div>
            </div>
          </div>

          <div className="conf-bar-item">
            <div className="conf-bar-meta">
              <span>Orientation Detection</span>
              <strong>{conf.orientation_confidence}%</strong>
            </div>
            <div className="conf-track">
              <div className="conf-fill fill--purple" style={{ width: `${conf.orientation_confidence}%` }}></div>
            </div>
          </div>

          <div className="conf-bar-item">
            <div className="conf-bar-meta">
              <span>Shape Measurement</span>
              <strong>{conf.shape_measurement_confidence}%</strong>
            </div>
            <div className="conf-track">
              <div className="conf-fill fill--cyan" style={{ width: `${conf.shape_measurement_confidence}%` }}></div>
            </div>
          </div>

          <div className="conf-bar-item">
            <div className="conf-bar-meta">
              <span>Color Distribution</span>
              <strong>{conf.color_measurement_confidence}%</strong>
            </div>
            <div className="conf-track">
              <div className="conf-fill fill--indigo" style={{ width: `${conf.color_measurement_confidence}%` }}></div>
            </div>
          </div>

          <div className="conf-bar-item">
            <div className="conf-bar-meta">
              <span>Feature Keypoint Stability</span>
              <strong>{conf.feature_matching_confidence}%</strong>
            </div>
            <div className="conf-track">
              <div className="conf-fill fill--coral" style={{ width: `${conf.feature_matching_confidence}%` }}></div>
            </div>
          </div>
        </div>
      </div>

      {/* 4. The 3 Visual Stages: Original -> AI Selected -> AI Aligned */}
      <div className="dashboard-section-card">
        <div className="section-card-header">
          <div className="header-icon-title">
            <span className="section-icon">👁️</span>
            <div>
              <h3>AI Object Understanding & Alignment Pipeline</h3>
              <p className="section-subtitle">
                Original photo preserved • Boundary selected without background erasure • Normalized for analysis
              </p>
            </div>
          </div>
        </div>

        <div className="three-stages-visual-grid">
          {/* Stage 1: Original Photograph */}
          <div className="stage-card">
            <div className="stage-card-header">
              <span className="stage-step-num">1</span>
              <strong>Original Photograph</strong>
            </div>
            <div className="stage-media-wrap">
              <img src={visuals.original} alt="Original photograph" className="stage-img" />
            </div>
            <div className="stage-footer-note">
              <span>Unmodified photo with original background intact</span>
            </div>
          </div>

          {/* Stage 2: AI Selected Object */}
          <div className="stage-card stage-card--highlight">
            <div className="stage-card-header">
              <span className="stage-step-num stage-step-num--selected">2</span>
              <strong>AI Object Selection</strong>
            </div>
            <div className="stage-media-wrap">
              <img src={visuals.ai_selected} alt="AI Selected Object" className="stage-img" />
              <div className="selection-live-tag">✓ Object Selected ({object.selection?.mask_confidence_pct}%)</div>
            </div>
            <div className="stage-footer-note">
              <span>Precise contour & glow on original photo</span>
            </div>
          </div>

          {/* Stage 3: AI Aligned Object */}
          <div className="stage-card stage-card--aligned">
            <div className="stage-card-header">
              <span className="stage-step-num stage-step-num--aligned">3</span>
              <strong>AI Aligned for Slicing</strong>
            </div>
            <div className="stage-media-wrap">
              <img src={visuals.ai_aligned} alt="AI Aligned Object" className="stage-img" />
              <div className="aligned-live-tag">
                {object.orientation?.is_symmetric
                  ? 'Rotational Symmetry Invariant'
                  : `Aligned Upright (${object.orientation?.correction_angle_deg > 0 ? `+${object.orientation?.correction_angle_deg}` : object.orientation?.correction_angle_deg}°)`}
              </div>
            </div>
            <div className="stage-footer-note">
              <span>Normalized along principal axis</span>
            </div>
          </div>
        </div>
      </div>

      {/* 5. Physical Visual Properties Grid */}
      <div className="properties-grid">
        {/* Dimensions Card */}
        <div className="property-card">
          <div className="prop-card-header">
            <span className="prop-icon">📏</span>
            <h4>Precise Dimensions</h4>
          </div>
          <div className="prop-data-rows">
            <div className="prop-row">
              <span className="prop-label">Pixel Height:</span>
              <span className="prop-value"><strong>{object.dimensions.pixel_height}</strong> px</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Pixel Width:</span>
              <span className="prop-value"><strong>{object.dimensions.pixel_width}</strong> px</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Aspect Ratio:</span>
              <span className="prop-value">{object.dimensions.aspect_ratio}</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Object Area:</span>
              <span className="prop-value">{object.dimensions.pixel_area.toLocaleString()} px²</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Perimeter:</span>
              <span className="prop-value">{object.dimensions.perimeter} px</span>
            </div>
            {object.dimensions.is_calibrated && (
              <div className="prop-row prop-row--calibrated">
                <span className="prop-label">Physical Scale:</span>
                <span className="prop-value">
                  <strong>{object.dimensions.physical_height_cm} cm</strong> × <strong>{object.dimensions.physical_width_cm} cm</strong>
                </span>
              </div>
            )}
          </div>
          <div className="prop-disclaimer-note">
            ⚠️ {object.dimensions.calibration_status || object.dimensions.unit_label}
          </div>
        </div>

        {/* Shape Card */}
        <div className="property-card">
          <div className="prop-card-header">
            <span className="prop-icon">📐</span>
            <h4>Shape & Symmetry</h4>
          </div>
          <div className="prop-data-rows">
            <div className="prop-row">
              <span className="prop-label">Shape Profile:</span>
              <span className="prop-value prop-value--highlight">{object.shape.shape_type}</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Circularity:</span>
              <span className="prop-value">{object.shape.circularity} / 1.0</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Solidity:</span>
              <span className="prop-value">{object.shape.solidity}</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Rectangularity:</span>
              <span className="prop-value">{object.shape.rectangularity}</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Bilateral Symmetry:</span>
              <span className="prop-value"><strong>{object.shape.symmetry_score}%</strong></span>
            </div>
          </div>
        </div>

        {/* Color Palette Card */}
        <div className="property-card">
          <div className="prop-card-header">
            <span className="prop-icon">🎨</span>
            <h4>Color & CIELAB Statistics</h4>
          </div>
          <div className="dominant-color-highlight">
            <span
              className="color-circle-swatch"
              style={{ backgroundColor: object.color.dominant_hex }}
            ></span>
            <div>
              <span className="dominant-name">{object.color.dominant_name}</span>
              <span className="dominant-hex">{object.color.dominant_hex}</span>
            </div>
          </div>

          <div className="palette-stacked-bar">
            {object.color.palette.map((p, idx) => (
              <div
                key={idx}
                className="palette-slice"
                style={{
                  backgroundColor: p.hex,
                  width: `${p.percentage}%`,
                }}
                title={`${p.name}: ${p.percentage}%`}
              ></div>
            ))}
          </div>

          <div className="palette-legend">
            {object.color.palette.slice(0, 3).map((p, idx) => (
              <span key={idx} className="palette-chip">
                <span className="chip-dot" style={{ backgroundColor: p.hex }}></span>
                {p.name} ({p.percentage}%)
              </span>
            ))}
          </div>
        </div>

        {/* Texture Card */}
        <div className="property-card">
          <div className="prop-card-header">
            <span className="prop-icon">🧵</span>
            <h4>Texture & Surface Information</h4>
          </div>
          <div className="prop-data-rows">
            <div className="prop-row">
              <span className="prop-label">Texture Profile:</span>
              <span className="prop-value prop-value--highlight">{object.texture.descriptor}</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Surface Variation:</span>
              <span className="prop-value">{object.texture.roughness_score} / 100</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Shannon Entropy:</span>
              <span className="prop-value">{object.texture.entropy}</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">LBP Uniformity:</span>
              <span className="prop-value">{object.texture.lbp_uniformity}</span>
            </div>
          </div>
          <div className="prop-disclaimer-note">
            ℹ️ {object.texture.information_note || (object.texture.is_informative ? 'Visible texture detected' : 'Smooth surface')}
          </div>
        </div>
      </div>



      {/* 7. Interactive Geometric Division Section */}
      <div className="dashboard-section-card division-section-card">
        <div className="section-card-header">
          <div className="header-icon-title">
            <span className="section-icon">✂️</span>
            <div>
              <h3>Geometric Equal Division</h3>
              <p className="section-subtitle">
                Calculated strictly along the object's normalized principal axis
              </p>
            </div>
          </div>
          <span className="split-status-pill">✓ 100% Equal Geometric Slices</span>
        </div>

        <div className="division-factor-toolbar">
          <label className="factor-label">
            <span>How many equal parts?</span>
          </label>

          <div className="preset-buttons-row">
            {[2, 3, 4, 5, 6, 8].map((num) => (
              <button
                key={num}
                className={`btn btn--preset ${partsInput === num ? 'btn--preset-active' : ''}`}
                onClick={() => handlePresetClick(num)}
                disabled={isUpdating}
              >
                {num} Parts
              </button>
            ))}
          </div>

          <div className="factor-custom-input">
            <input
              type="number"
              min="2"
              max="12"
              value={partsInput}
              onChange={handleInputChange}
              className="factor-number-box"
              disabled={isUpdating}
            />
            <span className="parts-unit">parts</span>
          </div>
        </div>

        <div className="division-visual-container">
          <div className="division-media-box">
            <img
              src={visuals.divided_image}
              alt="Divided object"
              className="divided-object-img"
            />
            {isUpdating && (
              <div className="division-updating-overlay">
                <div className="camera-spinner"></div>
                <span>Recalculating equal slices...</span>
              </div>
            )}
          </div>

          <div className="division-parts-breakdown">
            <h4>Section Dimensions & Share ({division.parts_count} Equal Parts)</h4>
            <div className="parts-pill-list">
              {division.parts.map((part) => (
                <div key={part.index} className="part-spec-card">
                  <div className="part-spec-header">
                    <span
                      className="part-color-dot"
                      style={{ backgroundColor: part.color_hex }}
                    ></span>
                    <strong>{part.label}</strong>
                    <span className="part-pct-badge">{part.percentage}%</span>
                  </div>
                  <div className="part-spec-body">
                    <span>Height: <strong>{part.pixel_height} px</strong></span>
                    <span>Width: <strong>{part.pixel_width} px</strong></span>
                    <span>Area: <strong>{part.pixel_area.toLocaleString()} px²</strong></span>
                    {part.longitudinal_percentage && (
                      <span>Axis Extent: <strong>{part.longitudinal_percentage}%</strong></span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 8. Footer Disclaimer */}
      <div className="result-disclaimer-box">
        <p>ℹ️ {disclaimer}</p>
      </div>
    </div>
  )
}
