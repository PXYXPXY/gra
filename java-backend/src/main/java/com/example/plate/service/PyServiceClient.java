package com.example.plate.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Mono;

@Service
public class PyServiceClient {
    private final WebClient client;
    private final boolean mockEnabled;

    public PyServiceClient(@Value("${python.service.base-url:http://localhost:8000}") String baseUrl,
                           @Value("${python.service.mock:true}") boolean mockEnabled) {
        this.client = WebClient.builder()
                .baseUrl(baseUrl)
                .build();
        this.mockEnabled = mockEnabled;
    }

    public ResponseEntity<String> forward(MultipartFile file) {
        if (mockEnabled) {
            return ResponseEntity.ok(mockPayload());
        }

        MultiValueMap<String, Object> formData = new LinkedMultiValueMap<>();
        formData.add("file", file.getResource());

        Mono<ResponseEntity<String>> response = client.post()
                .uri("/recognize")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(formData))
                .retrieve()
                .toEntity(String.class);

        return response.block();
    }

    private String mockPayload() {
        return "{\"text\":\"TEST123\",\"elapsedMs\":5,\"boxes\":[{\"x\":10,\"y\":12,\"w\":80,\"h\":30,\"score\":0.99}]}";
    }
}
