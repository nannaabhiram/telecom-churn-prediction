import {
  ADDON_OPTIONS,
  CONTRACT_OPTIONS,
  HELD_CONSTANT_SUMMARY,
  INTERNET_OPTIONS,
  MONTHLY_CHARGES_RANGE,
  PAYMENT_OPTIONS,
  PRESETS,
  TENURE_RANGE,
  hasInternet,
} from '../model/profile.js';

function RangeField({ id, label, value, range, format, onChange }) {
  const fill = ((value - range.min) / (range.max - range.min)) * 100;
  return (
    <div className="field">
      <div className="field-head">
        <label htmlFor={id}>{label}</label>
        <output htmlFor={id} className="field-value">
          {format(value)}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={range.min}
        max={range.max}
        step={range.step}
        value={value}
        style={{ '--fill': `${fill}%` }}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <div className="range-scale" aria-hidden="true">
        <span>{format(range.min)}</span>
        <span>{format(range.max)}</span>
      </div>
    </div>
  );
}

function SegmentedField({ name, legend, options, value, onChange, columns }) {
  return (
    <fieldset className="field">
      <legend>{legend}</legend>
      <div className="segmented" style={{ '--columns': columns ?? options.length }}>
        {options.map((option) => (
          <label key={option.value} className="segment">
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export default function ControlPanel({ profile, onChange, onPreset }) {
  const internet = hasInternet(profile);

  return (
    <section className="card panel" aria-labelledby="inputs-title">
      <header className="card-head">
        <p className="eyebrow">Step 1</p>
        <h2 id="inputs-title">Customer Parameter Inputs</h2>
      </header>

      <div className="presets" role="group" aria-label="Example customer profiles">
        <span className="presets-label">Try a profile</span>
        {PRESETS.map((preset) => (
          <button key={preset.id} type="button" className="chip-button" onClick={() => onPreset(preset.profile)}>
            {preset.label}
          </button>
        ))}
      </div>

      <RangeField
        id="tenure"
        label="Tenure"
        value={profile.tenure}
        range={TENURE_RANGE}
        format={(v) => `${v} mo`}
        onChange={(tenure) => onChange({ tenure })}
      />

      <RangeField
        id="monthly-charges"
        label="Monthly charges"
        value={profile.monthlyCharges}
        range={MONTHLY_CHARGES_RANGE}
        format={(v) => `$${v.toFixed(2)}`}
        onChange={(monthlyCharges) => onChange({ monthlyCharges })}
      />

      <SegmentedField
        name="contract"
        legend="Contract type"
        options={CONTRACT_OPTIONS}
        value={profile.contract}
        onChange={(contract) => onChange({ contract })}
      />

      <SegmentedField
        name="internet"
        legend="Internet service"
        options={INTERNET_OPTIONS}
        value={profile.internetService}
        onChange={(internetService) => onChange({ internetService })}
      />

      <SegmentedField
        name="payment"
        legend="Payment method"
        options={PAYMENT_OPTIONS}
        value={profile.paymentMethod}
        columns={2}
        onChange={(paymentMethod) => onChange({ paymentMethod })}
      />

      <fieldset className="field">
        <legend>Add-ons &amp; billing</legend>
        <div className="toggles">
          {ADDON_OPTIONS.map((addon) => {
            const disabled = addon.requiresInternet && !internet;
            return (
              <label key={addon.key} className={`toggle${disabled ? ' is-disabled' : ''}`}>
                <input
                  type="checkbox"
                  checked={profile[addon.key] && !disabled}
                  disabled={disabled}
                  onChange={(event) => onChange({ [addon.key]: event.target.checked })}
                />
                <span className="toggle-box" aria-hidden="true" />
                <span className="toggle-text">
                  {addon.label}
                  {disabled && <small>Requires internet</small>}
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>

      <p className="footnote">
        <strong>Held at the most common training value:</strong> {HELD_CONSTANT_SUMMARY.join(' · ')}.
      </p>
    </section>
  );
}
