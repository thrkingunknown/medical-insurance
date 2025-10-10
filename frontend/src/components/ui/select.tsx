import ReactSelect from "react-select";
import type { StylesConfig, GroupBase } from "react-select";
import { useTheme } from "../ThemeProvider";

interface Option {
  value: string;
  label: string;
}

interface SelectProps {
  id?: string;
  name?: string;
  value: string;
  onChange: (value: string) => void;
  options: Option[];
  placeholder?: string;
  className?: string;
  hasError?: boolean;
}

export function Select({
  id,
  name,
  value,
  onChange,
  options,
  placeholder,
  className,
  hasError,
}: SelectProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const selectedOption = options.find((opt) => opt.value === value) || null;

  const customStyles: StylesConfig<Option, false, GroupBase<Option>> = {
    control: (base, state) => ({
      ...base,
      minHeight: "40px",
      borderRadius: "0.5rem",
      borderColor: hasError
        ? "#ef4444"
        : state.isFocused
        ? isDark
          ? "#6b7280"
          : "#9ca3af"
        : isDark
        ? "#374151"
        : "#e5e7eb",
      backgroundColor: isDark ? "#374151" : "white",
      boxShadow: state.isFocused
        ? `0 0 0 2px ${isDark ? "#1f2937" : "#f3f4f6"}`
        : "none",
      cursor: "pointer",
      transition: "all 0.2s",
      "&:hover": {
        borderColor: isDark ? "#6b7280" : "#9ca3af",
        transform: "scale(1.01)",
      },
    }),
    menu: (base) => ({
      ...base,
      borderRadius: "0.75rem",
      overflow: "hidden",
      backgroundColor: isDark ? "#374151" : "white",
      boxShadow: isDark
        ? "0 10px 15px -3px rgba(0, 0, 0, 0.5)"
        : "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
      border: `1px solid ${isDark ? "#4b5563" : "#e5e7eb"}`,
    }),
    menuList: (base) => ({
      ...base,
      padding: "0.5rem",
      borderRadius: "0.75rem",
    }),
    option: (base, state) => ({
      ...base,
      borderRadius: "0.5rem",
      margin: "0.125rem 0",
      padding: "0.5rem 0.75rem",
      cursor: "pointer",
      backgroundColor: state.isSelected
        ? "#3b82f6"
        : state.isFocused
        ? isDark
          ? "#4b5563"
          : "#f3f4f6"
        : "transparent",
      color: state.isSelected ? "white" : isDark ? "#f3f4f6" : "#1f2937",
      transition: "all 0.15s",
      "&:active": {
        backgroundColor: "#3b82f6",
      },
    }),
    singleValue: (base) => ({
      ...base,
      color: isDark ? "#f3f4f6" : "#1f2937",
    }),
    placeholder: (base) => ({
      ...base,
      color: isDark ? "#9ca3af" : "#6b7280",
    }),
    dropdownIndicator: (base, state) => ({
      ...base,
      color: isDark ? "#9ca3af" : "#6b7280",
      transition: "all 0.2s",
      transform: state.selectProps.menuIsOpen ? "rotate(180deg)" : "rotate(0)",
      "&:hover": {
        color: isDark ? "#d1d5db" : "#4b5563",
      },
    }),
    indicatorSeparator: () => ({
      display: "none",
    }),
  };

  return (
    <ReactSelect
      id={id}
      name={name}
      value={selectedOption}
      onChange={(option) => onChange(option?.value || "")}
      options={options}
      placeholder={placeholder}
      className={className}
      styles={customStyles}
      isSearchable={false}
      components={{
        IndicatorSeparator: () => null,
      }}
    />
  );
}
