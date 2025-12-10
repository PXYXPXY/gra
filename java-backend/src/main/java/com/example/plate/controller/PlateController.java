package com.example.plate.controller;

import com.example.plate.service.PyServiceClient;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api/plate")
public class PlateController {
    private final PyServiceClient pyServiceClient;

    public PlateController(PyServiceClient pyServiceClient) {
        this.pyServiceClient = pyServiceClient;
    }

    @PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> recognize(@RequestParam("file") MultipartFile file) throws IOException {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body("{\"error\":\"file is empty\"}");
        }
        return pyServiceClient.forward(file);
    }
}
