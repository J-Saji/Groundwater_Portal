import { COLOR_PALETTE } from './constants';

// Extended color palette for aquifer types
const AQUIFER_COLORS = [
  "#E91E63", // Pink/Magenta
  "#1E3A8A", // Dark Blue
  "#7C3AED", // Purple
  "#3B82F6", // Light Blue
  "#DC2626", // Red
  "#10B981", // Green
  "#F59E0B", // Amber
  "#8B5CF6", // Violet
  "#06B6D4", // Cyan
  "#EC4899", // Pink
  "#14B8A6", // Teal
  "#F97316", // Orange
  "#6366F1", // Indigo
  "#84CC16", // Lime
  "#D946EF", // Fuchsia
  "#EF4444", // Bright Red
  "#22D3EE", // Sky Blue
  "#A855F7", // Purple Variant
];

/**
 * Simple string hash function
 * Converts a string to a consistent number
 */
const hashString = (str: string): number => {
  let hash = 0;
  const normalized = str.toLowerCase().trim();

  for (let i = 0; i < normalized.length; i++) {
    const char = normalized.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }

  return Math.abs(hash);
};

/**
 * Dynamically assigns colors to aquifer types
 * Same aquifer type = same color, always!
 */
export const getAquiferColor = (aquiferType: string, index: number): string => {
  if (!aquiferType) return "#9E9E9E"; // Gray for undefined

  // Generate consistent hash from aquifer type name
  const hash = hashString(aquiferType);

  // Map hash to color palette
  const colorIndex = hash % AQUIFER_COLORS.length;

  return AQUIFER_COLORS[colorIndex];
};

export const getGraceColor = (value: number): string => {
  if (value < -10) return "#8B0000";
  if (value < -5) return "#DC143C";
  if (value < 0) return "#FF6347";
  if (value < 5) return "#32CD32";
  if (value < 10) return "#1E90FF";
  return "#0000CD";
};

export const getRainfallColor = (value: number): string => {
  if (value < 1) return "#F0F0F0";
  if (value < 10) return "#B3E5FC";
  if (value < 25) return "#4FC3F7";
  if (value < 50) return "#2196F3";
  if (value < 100) return "#1976D2";
  return "#0D47A1";
};

export const getWellColor = (category: string): string => {
  switch (category) {
    case "Recharge": return "#1E88E5";
    case "Shallow (0-30m)": return "#43A047";
    case "Moderate (30-60m)": return "#FB8C00";
    case "Deep (60-100m)": return "#E53935";
    case "Very Deep (>100m)": return "#B71C1C";
    default: return "#9E9E9E";
  }
};

export const getStressColor = (category: string): string => {
  switch (category) {
    case 'Critical': return '#DC2626';
    case 'Stressed': return '#F59E0B';
    case 'Moderate': return '#FCD34D';
    case 'Healthy': return '#22C55E';
    default: return '#9CA3AF';
  }
};

export const getASIColor = (score: number): string => {
  if (score < 1) return "#FEE5D9";
  if (score < 2) return "#FCBBA1";
  if (score < 3) return "#FB6A4A";
  if (score < 4) return "#CB181D";
  return "#67000D";
};

export const getSASSColor = (score: number): string => {
  if (score < -1) return "#22C55E";
  if (score < 0) return "#84CC16";
  if (score < 1) return "#FCD34D";
  if (score < 2) return "#F59E0B";
  return "#DC2626";
};

export const getDensityColor = (density: number): string => {
  if (density < 5) return "#FEF0D9";
  if (density < 10) return "#FDCC8A";
  if (density < 20) return "#FC8D59";
  if (density < 40) return "#E34A33";
  return "#B30000";
};

export const getDivergenceColor = (value: number): string => {
  const numValue = Number(value);

  if (numValue < -2) return "#DC2626";
  if (numValue < -1) return "#F87171";
  if (numValue < 0) return "#FCA5A5";
  if (numValue < 1) return "#93C5FD";
  if (numValue < 2) return "#3B82F6";
  return "#1D4ED8";
};

export const getHotspotColor = (cluster: number): string => {
  if (cluster === -1) return "#9CA3AF";
  return COLOR_PALETTE[cluster % COLOR_PALETTE.length];
};

export const getTrendColor = (slope: number): string => {
  if (slope > 0) return "#DC2626";
  return "#16A34A";
};

export const getGWRColor = (resourceMCM: number, zmin: number, zmax: number): string => {
  const range = zmax - zmin;
  if (range < 0.001) return "#78C679";

  const normalized = Math.max(0, Math.min(1, (resourceMCM - zmin) / range));

  if (normalized < 0.1) return "#FFFFCC";
  if (normalized < 0.2) return "#C2E699";
  if (normalized < 0.4) return "#78C679";
  if (normalized < 0.6) return "#31A354";
  if (normalized < 0.8) return "#006837";
  return "#004529";
};