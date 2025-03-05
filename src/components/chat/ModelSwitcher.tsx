"use client";

import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ModelSwitcherProps {
  models: string[];
  currentModel: string;
  onModelChange: (model: string) => void;
  disabled?: boolean;
}

const ModelSwitcher: React.FC<ModelSwitcherProps> = ({
  models,
  currentModel,
  onModelChange,
  disabled = false,
}) => {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm font-medium">Model:</span>
      <Select
        value={currentModel}
        onValueChange={onModelChange}
        disabled={disabled}
      >
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Select model" />
        </SelectTrigger>
        <SelectContent>
          {models.map((model) => (
            <SelectItem key={model} value={model}>
              {model}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default ModelSwitcher;
